"""
IoV-BERT-IDS: the complete experimental pipeline of the paper in one command.

  python run_pipeline.py --data datasets                 # real downloaded datasets
  python run_pipeline.py --data datasets                 # or the output of generate_datasets.py
  python run_pipeline.py --data datasets --minutes 30    # larger time budget -> better models

Expected dataset layout:
  <data>/cicids/       *.pcap  + labelled-flow *.csv (CICIDS2017 GeneratedLabelledFlows)
  <data>/botiot/       <ClassName>/*.pcap            (one folder per class)
  <data>/car_hacking/  DoS_dataset.csv Fuzzy_dataset.csv gear_dataset.csv RPM_dataset.csv normal_run_data.txt
  <data>/ivn/          Sonata/*.csv Soul/*.csv Spark/*.csv   (IVN challenge, R/T Class column)
  <data>/pretrain/     optional unlabelled *.pcap(ng) for pre-training
                       (default: the pcaps in cicids/ and botiot/, used without labels)

Stages (as in the paper)
  1. Semantic Extractor   BSE -> BSL (pre-training), USE -> USL (fine-tuning), per-class sampling
  2. Pre-training         MBWM + NBSP on the BSL
  3. Fine-tuning          IoV-BERT-IDS on every experiment
  4. Baselines            AE, VAE, ByteSGAN on the same train/test split
  5. Evaluation           Precision, Recall, F1, Accuracy (Eqs. 11-14), confusion matrices
  6. Report               Tables I-VIII, Figs. 3-9, report.md in <out>/

Experiments
  Exp 1  extra-vehicle : CICIDS, BoT-IoT
  Exp 2  in-vehicle    : IVN-IDS (all vehicles), Car-Hacking
  Exp 3  generalisation: train HYUNDAI Sonata -> test KIA Soul
"""
import argparse
import glob
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

from data import (CLS, MASK, N_SPECIAL, PAD, SEP, encode, load_bsl, load_usl,
                  make_pair_example)

PY = sys.executable
MODELS = ["AE", "VAE", "ByteSGAN", "IoV-BERT-IDS"]
# fixed colour per model (validated categorical palette; colour follows the model in every figure)
COLORS = {"AE": "#2a78d6", "VAE": "#eb6834", "ByteSGAN": "#1baf7a", "IoV-BERT-IDS": "#eda100"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e2dc"
EXCLUDE_CICIDS = {"Infiltration", "Web Attack - Sql Injection", "Heartbleed"}   # excluded in the paper (Table I)

EXPS = [
    dict(key="exp1_cicids", exp="Exp 1", name="CICIDS", train="cicids.tsv"),
    dict(key="exp1_botiot", exp="Exp 1", name="BoT-IoT", train="botiot.tsv"),
    dict(key="exp2_ivn", exp="Exp 2", name="IVN-IDS", train="ivn.tsv"),
    dict(key="exp2_carhack", exp="Exp 2", name="Car-Hacking", train="carhack.tsv"),
    dict(key="exp3_cross", exp="Exp 3", name="IVN_Exp3", train="sonata.tsv", test="soul.tsv"),
]


# ======================================================================================
# helpers
# ======================================================================================
class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()
        self.problems = []

    def __call__(self, msg):
        line = f"[{(time.time() - self.t0) / 60:5.1f} min] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()


def run(cmd, log):
    log("$ " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        log.problems.append(" ".join(cmd[1:3]))
        log(f"   !! step failed (exit {r.returncode}) - continuing with the rest of the pipeline")
    return r.returncode == 0


def find_dir(root, *names):
    for n in names:
        for d in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(d) and os.path.basename(d).lower() == n.lower():
                return d
    return None


def pcaps_in(path):
    out = []
    for ext in ("*.pcap", "*.pcapng", "*.cap"):
        out += glob.glob(os.path.join(path, "**", ext), recursive=True)
    return sorted(out)


def order(names):
    """Normal / BENIGN class first, the rest alphabetically (like the paper's tables)."""
    first = [n for n in names if n.lower() in ("normal", "benign", "attack free", "attack-free")]
    return first + [n for n in names if n not in first]


def metrics(y_true, y_pred, names):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": float(accuracy_score(y_true, y_pred)), "per_class": {},
           "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "names": list(names)}
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[avg] = {"precision": float(pa), "recall": float(ra), "f1": float(fa)}
    for i, n in enumerate(names):
        out["per_class"][n] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
    return out


def filter_usl(path, exclude):
    rows = [l for l in open(path, encoding="utf-8") if l.split("\t", 1)[0] not in exclude]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    cpath = path + ".counts.json"
    if os.path.exists(cpath):
        c = json.load(open(cpath))
        json.dump({k: v for k, v in c.items() if k not in exclude}, open(cpath, "w"), indent=1)


# ======================================================================================
# 1. Semantic Extractor
# ======================================================================================
def stage_extract(a, D, U, log):
    os.makedirs(U, exist_ok=True)
    N = str(a.per_class)
    done = lambda f: a.reuse and os.path.exists(os.path.join(U, f))
    have = {}

    # ---- pre-training corpus (unlabelled) ----
    pre_dir = find_dir(D, "pretrain")
    pre_src = pcaps_in(pre_dir) if pre_dir else []
    if not pre_src:
        for n in ("cicids", "botiot"):
            d = find_dir(D, n)
            pre_src += pcaps_in(d) if d else []
    if pre_src and not done("bsl.tsv"):
        run([PY, "semantic_extractor.py", "pretrain", *pre_src, "--out", f"{U}/bsl.tsv",
             "--max-pairs", str(a.max_pairs), "--mode", a.traffic_mode], log)
    have["bsl"] = os.path.exists(f"{U}/bsl.tsv")

    # ---- Exp 1: CICIDS ----
    d = find_dir(D, "cicids", "cicids2017", "cic-ids2017")
    if d:
        csvs = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        if csvs and not done("cicids.tsv"):
            if run([PY, "semantic_extractor.py", "traffic", d, "--flows-csv", *csvs, "--mode", a.traffic_mode,
                    "--max-per-class", N, "--out", f"{U}/cicids.tsv"], log):
                filter_usl(f"{U}/cicids.tsv", EXCLUDE_CICIDS)
        elif not csvs:
            log("CICIDS: no labelled-flow CSV found next to the pcaps - skipping")
    # ---- Exp 1: BoT-IoT ----
    d = find_dir(D, "botiot", "bot-iot")
    if d and not done("botiot.tsv"):
        run([PY, "semantic_extractor.py", "traffic", d, "--mode", a.traffic_mode,
             "--max-per-class", N, "--out", f"{U}/botiot.tsv"], log)
    # ---- Exp 2: Car-Hacking ----
    d = find_dir(D, "car_hacking", "car-hacking", "carhacking")
    if d and not done("carhack.tsv"):
        files = [f for f in glob.glob(os.path.join(d, "*")) if os.path.isfile(f) and f.lower().endswith((".csv", ".txt"))
                 and "readme" not in f.lower()]
        run([PY, "semantic_extractor.py", "can", *files, "--max-per-class", N, "--out", f"{U}/carhack.tsv"], log)
    # ---- Exp 2 / Exp 3: IVN-IDS ----
    d = find_dir(D, "ivn", "ivn-ids", "ivn_ids")
    if d:
        gen = ["--format", "generic", "--id-col", a.ivn_id_col, "--data-cols", a.ivn_data_cols,
               "--label-col", a.ivn_label_col]
        allf = sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        if allf and not done("ivn.tsv"):
            run([PY, "semantic_extractor.py", "can", *allf, *gen, "--max-per-class", N, "--out", f"{U}/ivn.tsv"], log)
        son, soul = find_dir(d, "sonata", "hyundai_sonata"), find_dir(d, "soul", "kia_soul")
        if son and soul:
            if not done("sonata.tsv"):
                run([PY, "semantic_extractor.py", "can", *sorted(glob.glob(os.path.join(son, "*.csv"))), *gen,
                     "--max-per-class", str(int(a.per_class * 0.8)), "--out", f"{U}/sonata.tsv"], log)
            if not done("soul.tsv"):
                run([PY, "semantic_extractor.py", "can", *sorted(glob.glob(os.path.join(soul, "*.csv"))), *gen,
                     "--max-per-class", str(int(a.per_class * 0.2)), "--seed", "1", "--out", f"{U}/soul.tsv"], log)
    for e in EXPS:
        e["available"] = os.path.exists(f"{U}/{e['train']}") and (not e.get("test") or os.path.exists(f"{U}/{e['test']}"))
        if not e["available"]:
            log(f"{e['exp']} {e['name']}: data not found - experiment skipped")
    return have


# ======================================================================================
# 2-3. Pre-training and IoV-BERT-IDS fine-tuning (pretrain.py / finetune.py)
# ======================================================================================
def stage_pretrain(a, U, R, log, minutes):
    if a.reuse and os.path.exists(f"{R}/pretrain/bert.pt"):
        log("pre-training: reusing existing model"); return True
    return run([PY, "pretrain.py", "--bsl", f"{U}/bsl.tsv", "--preset", a.preset, "--steps", "1000000",
                "--warmup", "100", "--lr", "1e-3" if a.preset == "tiny" else "5e-4", "--batch-size", "32",
                "--max-minutes", f"{minutes:.2f}", "--log-every", "20", "--save-every", "300",
                "--out", f"{R}/pretrain"], log)


def stage_finetune(a, e, U, R, log, minutes, bert_ok):
    out = f"{R}/{e['key']}/IoV-BERT-IDS"
    if a.reuse and os.path.exists(f"{out}/predictions.npz"):
        return True
    cmd = [PY, "finetune.py", "--usl", f"{U}/{e['train']}", "--epochs", "200", "--batch-size", "64",
           "--lr", "5e-4", "--max-minutes", f"{minutes:.2f}", "--out", out]
    if e.get("test"):
        cmd += ["--test-usl", f"{U}/{e['test']}"]
    cmd += ["--bert", f"{R}/pretrain/bert.pt"] if bert_ok else ["--no-pretrain", "--preset", a.preset]
    return run(cmd, log)


# ======================================================================================
# 4. Baselines: AE, VAE, ByteSGAN (re-implemented on raw bytes, same split as IoV-BERT-IDS)
# ======================================================================================
def sentence_bytes(s, L):
    ws = s.split()
    if not ws:
        return np.zeros(L, np.float32)
    hx = ws[0] + "".join(w[2:] for w in ws[1:])
    try:
        b = bytes.fromhex(hx[: 2 * L])
    except ValueError:
        b = b""
    v = np.zeros(L, np.float32)
    v[: len(b)] = np.frombuffer(b, np.uint8) / 255.0
    return v


def load_experiment(e, U):
    labels, _, sents = load_usl(f"{U}/{e['train']}")
    if e.get("test"):
        tl, _, ts = load_usl(f"{U}/{e['test']}")
        labels, sents = labels + tl, sents + ts
    return labels, sents


def get_split(e, U, R, labels, seed=0):
    """Re-use IoV-BERT-IDS's exact split; fall back to the same stratified procedure."""
    p = f"{R}/{e['key']}/IoV-BERT-IDS/predictions.npz"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return list(z["names"]), z["train_idx"], z["val_idx"], z["test_idx"]
    from sklearn.model_selection import train_test_split
    names = sorted(set(labels)); y = np.array([names.index(l) for l in labels])
    n_train = len(load_usl(f"{U}/{e['train']}")[0])
    idx = np.arange(n_train)
    if e.get("test"):
        tr, te = idx, np.arange(n_train, len(labels))
    else:
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y[idx])
    tr, va = train_test_split(tr, test_size=0.1, random_state=seed)
    return names, tr, va, te


def stage_baselines(a, e, U, R, log, seconds):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(os.cpu_count() or 1)
    from sklearn.metrics import f1_score

    labels, sents = load_experiment(e, U)
    names, tr, va, te = get_split(e, U, R, labels)
    lid = {n: i for i, n in enumerate(names)}
    y = np.array([lid[l] for l in labels], dtype=np.int64)
    L, K = a.baseline_bytes, len(names)
    X = np.stack([sentence_bytes(s, L) for s in sents])

    class AE(nn.Module):                      # autoencoder + classifier on the latent code
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(L, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU())
            s.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, L), nn.Sigmoid())
            s.clf = nn.Linear(32, K)

        def loss(s, x, t):
            z = s.enc(x)
            return F.cross_entropy(s.clf(z), t) + F.mse_loss(s.dec(z), x)

        def predict(s, x):
            return s.clf(s.enc(x))

    class VAE(nn.Module):                     # variational autoencoder + classifier on the mean
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Linear(L, 128), nn.ReLU())
            s.mu, s.lv = nn.Linear(128, 16), nn.Linear(128, 16)
            s.dec = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, L), nn.Sigmoid())
            s.clf = nn.Linear(16, K)

        def loss(s, x, t):
            h = s.enc(x); mu, lv = s.mu(h), s.lv(h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            kl = -0.5 * torch.mean(torch.sum(1 + lv - mu ** 2 - lv.exp(), 1))
            return F.cross_entropy(s.clf(mu), t) + F.mse_loss(s.dec(z), x) + 1e-3 * kl

        def predict(s, x):
            return s.clf(s.mu(s.enc(x)))

    class ByteSGAN(nn.Module):                # semi-supervised GAN on byte vectors (K-class discriminator)
        def __init__(s):
            super().__init__()
            s.G = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, L), nn.Sigmoid())
            s.feat = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.LeakyReLU(0.2),
                                   nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2), nn.Flatten(),
                                   nn.Linear(64 * ((L + 1) // 2), 128), nn.LeakyReLU(0.2))
            s.head = nn.Linear(128, K)

        def D(s, x):
            f = s.feat(x.unsqueeze(1))
            return s.head(f), f

        def predict(s, x):
            return s.D(x)[0]

    def train(model_name):
        torch.manual_seed(0); rng = np.random.default_rng(0)
        m = {"AE": AE, "VAE": VAE, "ByteSGAN": ByteSGAN}[model_name]()
        if model_name == "ByteSGAN":
            opt_d = torch.optim.Adam(list(m.feat.parameters()) + list(m.head.parameters()), 1e-3, betas=(0.5, 0.999))
            opt_g = torch.optim.Adam(m.G.parameters(), 1e-3, betas=(0.5, 0.999))
        else:
            opt = torch.optim.Adam(m.parameters(), 1e-3)
        Xt, yt = torch.tensor(X), torch.tensor(y)
        best, best_state, t0, ep = -1.0, None, time.time(), 0
        while time.time() - t0 < seconds and ep < a.baseline_epochs:
            ep += 1; m.train()
            perm = rng.permutation(tr)
            for s0 in range(0, len(perm), 128):
                b = torch.tensor(perm[s0:s0 + 128])
                x, t = Xt[b], yt[b]
                if model_name == "ByteSGAN":
                    z = torch.randn(len(b), 32)
                    fake = m.G(z)
                    lr_, _ = m.D(x); lf_, _ = m.D(fake.detach())
                    lse_r, lse_f = torch.logsumexp(lr_, 1), torch.logsumexp(lf_, 1)
                    loss_d = (F.cross_entropy(lr_, t) - torch.mean(lse_r - F.softplus(lse_r))
                              + torch.mean(F.softplus(lse_f)))
                    opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                    _, fr = m.D(x); _, ff = m.D(m.G(torch.randn(len(b), 32)))
                    loss_g = F.mse_loss(ff.mean(0), fr.detach().mean(0))     # feature matching
                    opt_g.zero_grad(); loss_g.backward(); opt_g.step()
                else:
                    loss = m.loss(x, t)
                    opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vp = m.predict(Xt[torch.tensor(va)]).argmax(1).numpy()
            f1 = f1_score(y[va], vp, average="macro")
            if f1 > best:
                best, best_state = f1, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(best_state); m.eval()
        t1 = time.time()
        with torch.no_grad():
            tp = m.predict(Xt[torch.tensor(te)]).argmax(1).numpy()
        ms = (time.time() - t1) * 1000 / max(len(te), 1)
        return tp, ep, ms

    for mn in ("AE", "VAE", "ByteSGAN"):
        out = f"{R}/{e['key']}/{mn}"
        if a.reuse and os.path.exists(f"{out}/predictions.npz"):
            continue
        os.makedirs(out, exist_ok=True)
        try:
            tp, ep, ms = train(mn)
        except Exception as ex:                                    # keep the pipeline going
            log(f"   !! {mn} on {e['name']} failed: {ex}"); log.problems.append(f"{mn} {e['name']}"); continue
        np.savez(f"{out}/predictions.npz", y_true=y[te], y_pred=tp, names=np.array(names),
                 test_idx=te, train_idx=tr, val_idx=va)
        res = metrics(y[te], tp, names); res.update(epochs=ep, inference_ms_per_sample=ms)
        json.dump(res, open(f"{out}/results.json", "w"), indent=1)
        log(f"   {mn:<9} {e['name']:<12} macro-F1 {res['macro']['f1']:.4f}  acc {res['accuracy']:.4f}  ({ep} epochs)")


# ======================================================================================
# 5. Collect results
# ======================================================================================
def collect(R):
    res = {}
    for e in EXPS:
        for m in MODELS:
            p = f"{R}/{e['key']}/{m}/predictions.npz"
            if os.path.exists(p):
                z = np.load(p, allow_pickle=False)
                r = metrics(z["y_true"], z["y_pred"], [str(n) for n in z["names"]])
                rj = f"{R}/{e['key']}/{m}/results.json"
                if os.path.exists(rj):
                    old = json.load(open(rj))
                    r["inference_ms_per_sample"] = old.get("inference_ms_per_sample")
                    r["history"] = old.get("history")
                res[(e["key"], m)] = r
    return res


# ======================================================================================
# 6. Tables and figures
# ======================================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
    return plt


def write_table(path_noext, title, header, rows, note="", bold_best_cols=()):
    """Write a table as .csv, .md and .png. bold_best_cols: column indexes where the max is bolded."""
    import csv
    with open(path_noext + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    best = {}
    for c in bold_best_cols:
        vals = [r[c] for r in rows if isinstance(r[c], float)]
        best[c] = max(vals) if vals else None
    md = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        md.append("| " + " | ".join((f"**{fmt(v)}**" if c in best and v == best[c] else fmt(v))
                                    for c, v in enumerate(r)) + " |")
    if note:
        md += ["", f"*{note}*"]
    open(path_noext + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    plt = _plt()
    fig_h = 0.6 + 0.26 * (len(rows) + 2)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.0 * len(header) + 1), fig_h))
    ax.axis("off")
    cell = [[fmt(v) for v in r] for r in rows]
    # long headers ("CICIDS Precision") wrap onto two lines instead of being cut off
    hdr = [h.replace(" ", "\n", 1) if len(h) > 11 and " " in h else h for h in header]
    two = any("\n" in h for h in hdr)
    t = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#9a988f"); c.set_linewidth(0.5)
        if i == 0:
            c.set_text_props(weight="bold")
            if two:
                c.set_height(c.get_height() * 1.9)
        elif j in best and rows[i - 1][j] == best[j]:
            c.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10, pad=6)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color=INK2, wrap=True)
    fig.savefig(path_noext + ".png"); plt.close(fig)


def dataset_tables(U, T, synth):
    def counts(f):
        p = f"{U}/{f}.counts.json"
        return json.load(open(p)) if os.path.exists(p) else None
    note = "Original = samples present in the source files; Exp = samples used (per-class random sampling)."
    if synth:
        note += " Synthetic data."
    specs = [("table1_cicids_dataset", "TABLE I  CICIDS dataset", "cicids.tsv", "Exp 1"),
             ("table2_botiot_dataset", "TABLE II  BoT-IoT dataset", "botiot.tsv", "Exp 1"),
             ("table3_carhacking_dataset", "TABLE III  Car-Hacking dataset", "carhack.tsv", "Exp 2")]
    made = []
    for fn, title, f, ex in specs:
        c = counts(f)
        if c:
            rows = [[k, c[k]["original"], c[k]["used"]] for k in order(sorted(c))]
            write_table(f"{T}/{fn}", title, ["Classname", "Original (packets/frames)", ex], rows, note)
            made.append(fn)
    c2, cs, ck = counts("ivn.tsv"), counts("sonata.tsv"), counts("soul.tsv")
    if c2:
        rows = []
        for k in order(sorted(c2)):
            rows.append([k, c2[k]["original"], c2[k]["used"],
                         cs[k]["original"] if cs and k in cs else "-", cs[k]["used"] if cs and k in cs else "-",
                         ck[k]["original"] if ck and k in ck else "-", ck[k]["used"] if ck and k in ck else "-"])
        write_table(f"{T}/table4_ivn_dataset", "TABLE IV  IVN-IDS dataset",
                    ["Classname", "Original (all)", "Exp 2", "Sonata orig.", "Sonata Exp 3", "Soul orig.", "Soul Exp 3"],
                    rows, note)
        made.append("table4_ivn_dataset")
    return made


def environment_table(T, a):
    try:
        import torch
        tv, dev = torch.__version__, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
    except Exception:
        tv, dev = "not installed", "none (CPU)"
    ram = "n/a"
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2 ** 30:.0f} GB"
    except Exception:
        try:
            ram = f"{os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30:.0f} GB"
        except Exception:
            pass
    cpu = platform.processor() or platform.machine()
    rows = [["CPU", f"{cpu} ({os.cpu_count()} threads)"], ["RAM", ram], ["GPU", dev],
            ["OS", f"{platform.system()} {platform.release()}"], ["Python version", platform.python_version()],
            ["PyTorch version", tv], ["Model size", f"IoV-BERT '{a.preset}' preset"],
            ["Time budget", f"{a.minutes:g} min"]]
    write_table(f"{T}/table5_environment", "TABLE V  Equipment configuration", ["Name", "Specification"], rows)


def metric_tables(res, T, synth):
    note = "Macro-averaged Precision/Recall/F1 over classes. Bold = best in column."
    if synth:
        note += " Synthetic data."
    made = []
    for fn, title, keys in [("table6_exp1", "TABLE VI  Exp 1: evaluation metric comparison", ["exp1_cicids", "exp1_botiot"]),
                            ("table7_exp2", "TABLE VII  Exp 2: evaluation metric comparison", ["exp2_ivn", "exp2_carhack"])]:
        keys = [k for k in keys if any((k, m) in res for m in MODELS)]
        if not keys:
            continue
        dn = {e["key"]: e["name"] for e in EXPS}
        header = ["Model"] + [f"{dn[k]} {h}" for k in keys for h in ("Precision", "Recall", "F1", "ACC")]
        rows = []
        for m in MODELS:
            row = [m]
            for k in keys:
                r = res.get((k, m))
                row += ([r["macro"]["precision"], r["macro"]["recall"], r["macro"]["f1"], r["accuracy"]]
                        if r else ["-"] * 4)
            rows.append(row)
        write_table(f"{T}/{fn}", title, header, rows, note, bold_best_cols=range(1, len(header)))
        made.append(fn)
    if any(("exp3_cross", m) in res for m in MODELS):
        names = order(next(res[("exp3_cross", m)]["names"] for m in MODELS if ("exp3_cross", m) in res))
        header = ["Model"] + [f"{n} {h}" for n in names for h in ("P", "R", "F1")]
        rows = []
        for m in MODELS:
            r = res.get(("exp3_cross", m)); row = [m]
            for n in names:
                pc = r["per_class"][n] if r else None
                row += [pc["precision"], pc["recall"], pc["f1"]] if pc else ["-"] * 3
            rows.append(row)
        write_table(f"{T}/table8_exp3", "TABLE VIII  Exp 3: evaluation metric comparison (train Sonata, test Soul)",
                    header, rows, ("Per-class Precision/Recall/F1. Bold = best in column." + (" Synthetic data." if synth else "")),
                    bold_best_cols=range(1, len(header)))
        made.append("table8_exp3")
    return made


def suffix(synth):
    return "  [synthetic data]" if synth else ""


def fig_confusion(res, F_, synth, keys, fname, title):
    plt = _plt()
    keys = [k for k in keys if any((k, m) in res for m in MODELS)]
    if not keys:
        return None
    dn = {e["key"]: e["name"] for e in EXPS}
    nc = max(len(res[(k, m)]["names"]) for k in keys for m in MODELS if (k, m) in res)
    cs = 2.2 + 0.28 * nc
    fig, axes = plt.subplots(len(keys), len(MODELS), figsize=(cs * len(MODELS), cs * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        for j, m in enumerate(MODELS):
            ax = axes[i][j]
            r = res.get((k, m))
            if not r:
                ax.axis("off"); ax.text(0.5, 0.5, "not run", ha="center", color=INK2); continue
            names = r["names"]; o = [names.index(n) for n in order(names)]
            cm = np.array(r["confusion_matrix"], float)[np.ix_(o, o)]
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            lab = [names[x] for x in o]; n = len(lab); fs = 7 if n <= 6 else 5
            ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=fs)
            ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=fs)
            for a_ in range(n):
                for b_ in range(n):
                    v = cmn[a_, b_]
                    if v >= 0.005 or a_ == b_:
                        ax.text(b_, a_, f"{v:.2f}", ha="center", va="center", fontsize=fs - 0.5,
                                color="white" if v > 0.55 else INK)
            ax.set_title(f"{dn[k]} - {m}", fontsize=9)
            if j == 0:
                ax.set_ylabel("True label", fontsize=8)
            if i == len(keys) - 1:
                ax.set_xlabel("Predicted label", fontsize=8)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.suptitle(title + suffix(synth), fontsize=11, y=1.0)
    fig.tight_layout()
    path = f"{F_}/{fname}.png"; fig.savefig(path); plt.close(fig)
    return path


def fig_radar(res, F_, synth):
    plt = _plt()
    keys = [k for k in ("exp1_cicids", "exp1_botiot") if any((k, m) in res for m in MODELS)]
    if not keys:
        return None
    dn = {e["key"]: e["name"] for e in EXPS}
    fig = plt.figure(figsize=(6.2 * len(keys), 6.0))
    for i, k in enumerate(keys):
        ax = fig.add_subplot(1, len(keys), i + 1, polar=True)
        names = order(next(res[(k, m)]["names"] for m in MODELS if (k, m) in res))
        ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
        for m in MODELS:
            r = res.get((k, m))
            if not r:
                continue
            v = [r["per_class"][n]["precision"] for n in names]
            ax.plot(ang + ang[:1], v + v[:1], color=COLORS[m], lw=2.4 if m == "IoV-BERT-IDS" else 1.5,
                    marker="o", ms=3, label=m, zorder=3 if m == "IoV-BERT-IDS" else 2)
            if m == "IoV-BERT-IDS":
                ax.fill(ang + ang[:1], v + v[:1], color=COLORS[m], alpha=0.18)
        ax.set_xticks(ang); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
        ax.grid(color=GRID, lw=0.6); ax.spines["polar"].set_color(GRID)
        ax.set_title(f"({'ab'[i]}) {dn[k]}", fontsize=10, pad=22)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=9)
    fig.suptitle("Fig. 7  Classification precision comparison (per class)" + suffix(synth), fontsize=11, y=1.08)
    path = f"{F_}/fig7_precision_radar.png"; fig.savefig(path); plt.close(fig)
    return path


def _prf_axes(ax, res, key, names, kind):
    xs, xt, groups = [], [], []
    pos = 0.0
    for n in names:
        start = pos
        for met, lab in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
            xs.append(pos); xt.append(lab); pos += 1
        groups.append(((start + pos - 1) / 2, n)); pos += 0.6
    ms = [m for m in MODELS if (key, m) in res]
    w = 0.8 / max(len(ms), 1)
    for j, m in enumerate(ms):
        r = res[(key, m)]
        vals = [r["per_class"][n][met] for n in names for met in ("precision", "recall", "f1")]
        if kind == "bar":
            ax.bar(np.array(xs) + (j - (len(ms) - 1) / 2) * w, vals, width=w * 0.9, color=COLORS[m], label=m,
                   edgecolor="white", linewidth=0.4)
        else:
            seg = 3
            for g in range(len(names)):
                ax.plot(xs[g * seg:(g + 1) * seg], vals[g * seg:(g + 1) * seg], color=COLORS[m], lw=2,
                        marker="o", ms=5, label=m if g == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(xt, fontsize=7)
    for c, n in groups:
        ax.text(c, -0.13, n, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1.05); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13))


def fig_bars(res, F_, synth):
    plt = _plt()
    keys = [k for k in ("exp2_ivn", "exp2_carhack") if any((k, m) in res for m in MODELS)]
    if not keys:
        return None
    dn = {e["key"]: e["name"] for e in EXPS}
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3.6 * len(keys)), squeeze=False)
    for i, k in enumerate(keys):
        names = order(next(res[(k, m)]["names"] for m in MODELS if (k, m) in res))
        _prf_axes(axes[i][0], res, k, names, "bar")
        axes[i][0].set_title(f"({'ab'[i]}) {dn[k]}", fontsize=10, y=1.12)
    fig.suptitle("Fig. 8  Classification performance comparison: Precision, Recall, F1-score" + suffix(synth),
                 fontsize=11, y=1.01)
    fig.tight_layout()
    path = f"{F_}/fig8_prf_bars.png"; fig.savefig(path); plt.close(fig)
    return path


def fig_generalisation(res, F_, synth):
    plt = _plt()
    if not any(("exp3_cross", m) in res for m in MODELS):
        return None
    names = order(next(res[("exp3_cross", m)]["names"] for m in MODELS if ("exp3_cross", m) in res))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _prf_axes(ax, res, "exp3_cross", names, "line")
    fig.suptitle("Fig. 9  Classification generalisation in in-vehicle networks (train Sonata, test Soul)" + suffix(synth),
                 fontsize=11, y=1.06)
    path = f"{F_}/fig9_generalisation.png"; fig.savefig(path); plt.close(fig)
    return path


def fig_can_structure(D, F_, synth):
    """Fig. 5: feature structure of a CAN packet, with a real row from the data."""
    plt = _plt()
    sample = None
    d = find_dir(D, "car_hacking", "car-hacking", "carhacking")
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))) if d else []:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                p = line.strip().split(",")
                if len(p) >= 4 and p[2].isdigit():
                    dlc = int(p[2]); sample = (p[0], p[1], str(dlc), p[3:3 + dlc], p[3 + dlc] if len(p) > 3 + dlc else "")
                    break
        if sample:
            break
    fields = [("Timestamp", 2.2, "#dfeee0"), ("CAN ID", 1.2, "#cde9de"), ("DLC", 0.8, "#f6efd2"),
              ("DATA[0-7]", 5.6, "#e4ebf7"), ("Label", 1.2, "#f5d7cf")]
    fig, ax = plt.subplots(figsize=(10, 2.3 if sample else 1.4))
    x = 0
    for name, w, col in fields:
        ax.add_patch(plt.Rectangle((x, 1), w, 0.8, facecolor=col, edgecolor="#8a887f", lw=0.6))
        ax.text(x + w / 2, 1.4, name, ha="center", va="center", fontsize=10)
        if name == "DATA[0-7]":
            for k in range(1, 8):
                ax.plot([x + k * w / 8] * 2, [1, 1.8], color="#b4b2a9", lw=0.5, ls=":")
        if sample:
            val = {"Timestamp": sample[0], "CAN ID": sample[1], "DLC": sample[2],
                   "DATA[0-7]": " ".join(sample[3]), "Label": sample[4]}[name]
            ax.text(x + w / 2, 0.55, val, ha="center", va="center", fontsize=8.5, family="monospace", color=INK2)
        x += w
    if sample:
        cbs = (sample[1].zfill(4) + "".join(sample[3])).lower()
        words = " ".join(cbs[i:i + 4] for i in range(0, len(cbs) - 2, 2))
        ax.text(0, 0.05, f"CAN Byte Sentence (CAN ID + DATA, overlapping 2-byte words):  {words}",
                fontsize=8, family="monospace", color=INK)
    ax.set_xlim(-0.1, x + 0.1); ax.set_ylim(-0.1 if sample else 0.9, 1.9); ax.axis("off")
    ax.set_title("Fig. 5  Feature structure of CAN packets" + suffix(synth), fontsize=11)
    path = f"{F_}/fig5_can_structure.png"; fig.savefig(path); plt.close(fig)
    return path


def _tok(i):
    return {PAD: "[PAD]", 1: "[UNK]", CLS: "[CLS]", SEP: "[SEP]", MASK: "[MASK]"}.get(int(i), f"{int(i) - N_SPECIAL:04x}")


def fig_pretrain_examples(U, R, F_, synth):
    """Figs. 3 and 4: the MBWM and NBSP tasks on a real sample, with the pre-trained model's predictions."""
    import torch
    from model import IoVBertPretrain, MiniBert
    plt = _plt()
    ck = torch.load(f"{R}/pretrain/checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = torch.load(f"{R}/pretrain/bert.pt", map_location="cpu", weights_only=False)["config"]
    m = IoVBertPretrain(MiniBert(**cfg)); m.load_state_dict(ck["model"]); m.eval()
    pairs = [p for p in load_bsl(f"{U}/bsl.tsv", limit=5000) if len(p[0].split()) >= 8 and len(p[1].split()) >= 8]
    if len(pairs) < 2:
        return []
    rng = random.Random(3)
    a_s, b_s = pairs[rng.randrange(len(pairs))]
    a_ids, b_ids = encode(a_s)[:8], encode(b_s)[:8]
    ids, seg = make_pair_example(a_ids, b_ids, cfg["max_len"])

    def encode_t(ids_, seg_):
        t = torch.tensor([ids_]); s = torch.tensor([seg_])
        return m.bert(t, s, torch.zeros_like(t, dtype=torch.bool))

    # ---- Fig 3: MBWM ----
    real = [i for i, t in enumerate(ids) if t >= N_SPECIAL]
    pos = sorted(rng.sample(real, 3))
    masked = list(ids)
    for p_ in pos:
        masked[p_] = MASK
    with torch.no_grad():
        h = encode_t(masked, seg)
        logits = m.mlm_transform(h[0, pos]) @ m.bert.tok.weight.t() + m.mlm_bias
        prob = torch.softmax(logits, -1)
        top = prob.argmax(-1).tolist(); pmax = prob.max(-1).values.tolist()
    n = len(ids)
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * n), 2.6))
    for i in range(n):
        for row, toks, y in (("input", masked, 2), ("original", ids, 1)):
            t = toks[i]
            col = "#f6e2a8" if t == MASK else ("#e4ebf7" if t >= N_SPECIAL else "#ecebe6")
            ax.add_patch(plt.Rectangle((i * 0.62, y), 0.56, 0.5, facecolor=col, edgecolor="#8a887f", lw=0.5))
            ax.text(i * 0.62 + 0.28, y + 0.25, _tok(t), ha="center", va="center", fontsize=6.5, family="monospace")
    for k, p_ in enumerate(pos):
        ok = top[k] == ids[p_]
        ax.text(p_ * 0.62 + 0.28, 0.6, f"pred {_tok(top[k])}\n{pmax[k]:.2f} {'(correct)' if ok else '(wrong)'}",
                ha="center", va="center", fontsize=6.3, color=INK)
    ax.text(-0.1, 2.25, "masked input", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(-0.1, 1.25, "original", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(-0.1, 0.6, "MBWM prediction", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xlim(-2.2, n * 0.62); ax.set_ylim(0.2, 2.7); ax.axis("off")
    ax.set_title("Fig. 3  Masked Byte Word Model (MBWM) on a sample pair, with the pre-trained model's predictions"
                 + suffix(synth), fontsize=10)
    p3 = f"{F_}/fig3_mbwm_example.png"; fig.savefig(p3); plt.close(fig)

    # ---- Fig 4: NBSP ----
    other = pairs[rng.randrange(len(pairs))][1]
    exs = [("IsNext (real consecutive packets)", b_ids), ("NotNext (random packet)", encode(other)[:8])]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.62 * n), 2.9))
    for ax, (lab, bb) in zip(axes, exs):
        ids2, seg2 = make_pair_example(a_ids, bb, cfg["max_len"])
        with torch.no_grad():
            p_is = torch.softmax(m.nsp(m.pooler(encode_t(ids2, seg2))), -1)[0, 0].item()
        for i, t in enumerate(ids2):
            col = "#cde9de" if seg2[i] == 0 else "#f5d7cf"
            ax.add_patch(plt.Rectangle((i * 0.62, 0), 0.56, 0.5, facecolor=col if t >= N_SPECIAL else "#ecebe6",
                                       edgecolor="#8a887f", lw=0.5))
            ax.text(i * 0.62 + 0.28, 0.25, _tok(t), ha="center", va="center", fontsize=6.5, family="monospace")
        ax.text(len(ids2) * 0.62 + 0.2, 0.25, f"P(IsNext) = {p_is:.2f}", va="center", fontsize=8.5)
        ax.set_title(lab, fontsize=8.5, loc="left", color=INK2)
        ax.set_xlim(-0.1, len(ids2) * 0.62 + 2.4); ax.set_ylim(-0.1, 0.6); ax.axis("off")
    fig.suptitle("Fig. 4  Next Byte Sentence Prediction (NBSP): sentence A (green) + sentence B (red)" + suffix(synth),
                 fontsize=10)
    fig.tight_layout()
    p4 = f"{F_}/fig4_nbsp_example.png"; fig.savefig(p4); plt.close(fig)
    return [p3, p4]


def fig_training_curves(R, res, F_, synth):
    plt = _plt()
    out = []
    lp = f"{R}/pretrain/log.jsonl"
    if os.path.exists(lp):
        recs = [json.loads(l) for l in open(lp) if l.strip()]
        if recs:
            st = [r["step"] for r in recs]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
            a1.plot(st, [r["mbwm"] for r in recs], color="#6250d6", lw=2, label="MBWM loss")
            a1.plot(st, [r["nbsp"] for r in recs], color="#e87ba4", lw=2, ls="--", label="NBSP loss")
            a1.set_xlabel("step"); a1.set_title("Pre-training loss", fontsize=9); a1.legend()
            a2.plot(st, [r["mbwm_acc"] for r in recs], color="#6250d6", lw=2, label="MBWM accuracy")
            a2.plot(st, [r["nbsp_acc"] for r in recs], color="#e87ba4", lw=2, ls="--", label="NBSP accuracy")
            a2.set_ylim(0, 1); a2.set_xlabel("step"); a2.set_title("Pre-training accuracy", fontsize=9); a2.legend()
            for ax in (a1, a2):
                ax.yaxis.grid(True, color=GRID, lw=0.6)
            fig.suptitle("IoV-BERT pre-training (MBWM + NBSP)" + suffix(synth), fontsize=11)
            fig.tight_layout(); p = f"{F_}/fig_pretraining_curves.png"; fig.savefig(p); plt.close(fig); out.append(p)
    hs = [(e["name"], res[(e["key"], "IoV-BERT-IDS")].get("history")) for e in EXPS
          if (e["key"], "IoV-BERT-IDS") in res and res[(e["key"], "IoV-BERT-IDS")].get("history")]
    if hs:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        seq = ["#e87ba4", "#008300", "#6250d6", "#e34948", "#2a78d6"]     # validated; markers as 2nd cue
        for (nm, h), c, mk in zip(hs, seq, "osD^v"):
            ax.plot([x["epoch"] for x in h], [x["val_macro_f1"] for x in h], color=c, lw=2, marker=mk, ms=5, label=nm)
        ax.set_xlabel("epoch"); ax.set_ylabel("validation macro-F1"); ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=8, ncol=3)
        ax.set_title("IoV-BERT-IDS fine-tuning" + suffix(synth), fontsize=10)
        p = f"{F_}/fig_finetuning_curves.png"; fig.savefig(p); plt.close(fig); out.append(p)
    return out


def write_report(a, D, R, T, F_, res, figs, tables, synth, log, t_total):
    L = ["# IoV-BERT-IDS - experimental results", ""]
    if synth:
        L += ["> **Data source: synthetic data** generated by `generate_datasets.py`, modelled on the published "
              "specifications of CICIDS2017, BoT-IoT, Car-Hacking and the IVN challenge dataset. These results are "
              "not results on the original datasets.", ""]
    L += [f"- Data folder: `{os.path.abspath(D)}`",
          f"- Samples per class: {a.per_class} (Exp 3: {int(a.per_class * 0.8)} train / {int(a.per_class * 0.2)} test)",
          f"- Model: IoV-BERT `{a.preset}` preset; time budget {a.minutes:g} min; total run time {t_total:.1f} min",
          "- Metrics: Precision, Recall, F1 (macro average over classes) and Accuracy, Eqs. (11)-(14) of the paper",
          "- Baselines AE, VAE and ByteSGAN are re-implemented here and trained on the first "
          f"{a.baseline_bytes} raw bytes of each packet/frame, on exactly the same train/test split as IoV-BERT-IDS "
          "(the paper's baselines used flow-level features).", ""]
    if log.problems:
        L += ["**Steps that failed:** " + ", ".join(log.problems), ""]
    for t in tables:
        L += [open(f"{T}/{t}.md", encoding="utf-8").read(), ""]
    L += ["## Figures", ""]
    for p in figs:
        L += [f"![{os.path.basename(p)}](figures/{os.path.basename(p)})", ""]
    open(f"{R}/report.md", "w", encoding="utf-8").write("\n".join(L))


def stage_report(a, D, U, R, log, t_start, with_model_examples=True):
    synth = os.path.exists(os.path.join(D, "README_SYNTHETIC.txt"))
    T, F_ = f"{R}/tables", f"{R}/figures"
    os.makedirs(T, exist_ok=True); os.makedirs(F_, exist_ok=True)
    res = collect(R)
    tables = dataset_tables(U, T, synth)
    environment_table(T, a); tables.append("table5_environment")
    tables += metric_tables(res, T, synth)
    figs = []
    steps = [lambda: [fig_can_structure(D, F_, synth)],
             lambda: fig_pretrain_examples(U, R, F_, synth) if with_model_examples and os.path.exists(f"{R}/pretrain/checkpoint.pt") else [],
             lambda: fig_training_curves(R, res, F_, synth),
             lambda: [fig_confusion(res, F_, synth, ["exp1_cicids", "exp1_botiot", "exp2_ivn", "exp2_carhack"],
                                    "fig6_confusion_matrices", "Fig. 6  Confusion matrices of classification performance")],
             lambda: [fig_radar(res, F_, synth)],
             lambda: [fig_bars(res, F_, synth)],
             lambda: [fig_generalisation(res, F_, synth)],
             lambda: [fig_confusion(res, F_, synth, ["exp3_cross"], "fig9b_exp3_confusion_matrices",
                                    "Exp 3 confusion matrices (train Sonata, test Soul)")]]
    for s in steps:
        try:
            figs += [p for p in s() if p]
        except Exception as ex:
            log(f"   !! figure failed: {ex}"); log.problems.append("figure")
    order_key = lambda p: (os.path.basename(p).split("_")[0].replace("fig", "").zfill(3), p)
    figs = sorted(set(figs), key=order_key)
    write_report(a, D, R, T, F_, res, figs, tables, synth, log, (time.time() - t_start) / 60)
    return res, figs, tables


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets", help="dataset folder (see layout above)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--minutes", type=float, default=10.0, help="total time budget (default 10)")
    ap.add_argument("--per-class", type=int, default=1000, help="samples per class (paper: 10,000)")
    ap.add_argument("--preset", default="tiny", choices=["tiny", "small", "base"])
    ap.add_argument("--traffic-mode", default="payload", choices=["payload", "packet"],
                    help="payload = paper; packet = anonymised headers + payload")
    ap.add_argument("--max-pairs", type=int, default=50000, help="pre-training sentence pairs")
    ap.add_argument("--baseline-bytes", type=int, default=64)
    ap.add_argument("--baseline-epochs", type=int, default=60)
    ap.add_argument("--ivn-id-col", default="Arbitration_ID")
    ap.add_argument("--ivn-data-cols", default="Data")
    ap.add_argument("--ivn-label-col", default="Class")
    ap.add_argument("--reuse", action="store_true", help="skip stages whose outputs already exist")
    a = ap.parse_args()

    t_start = time.time()
    D, R = a.data, a.out
    U = f"{R}/data"
    os.makedirs(R, exist_ok=True)
    log = Log(f"{R}/pipeline.log")
    if not os.path.isdir(D):
        sys.exit(f"data folder '{D}' not found (see the layout in: python run_pipeline.py -h)")
    if os.path.exists(os.path.join(D, "README_SYNTHETIC.txt")):
        log("NOTE: synthetic data detected - all results will be labelled 'synthetic data'")

    # 1. Semantic Extractor
    log("== 1. Semantic Extractor ==")
    have = stage_extract(a, D, U, log)
    exps = [e for e in EXPS if e["available"]]
    if not exps:
        sys.exit("no experiment has data - check the --data folder layout")

    # time budget (whatever extraction used is taken out of the remaining budget)
    left = max(a.minutes - (time.time() - t_start) / 60, a.minutes * 0.5)
    pre_min = left * 0.33 if have["bsl"] else 0
    ft_min = left * 0.42 / len(exps)
    bl_sec = left * 0.13 * 60 / (3 * len(exps))

    # 2. Pre-training
    bert_ok = False
    if have["bsl"]:
        log(f"== 2. Pre-training (MBWM + NBSP), {pre_min:.1f} min ==")
        bert_ok = stage_pretrain(a, U, R, log, pre_min) and os.path.exists(f"{R}/pretrain/bert.pt")
    else:
        log("== 2. Pre-training skipped (no pcaps for the pre-training corpus) - fine-tuning from scratch ==")

    # 3-4. Fine-tuning + baselines per experiment
    for e in exps:
        log(f"== 3. {e['exp']} {e['name']}: fine-tuning IoV-BERT-IDS ({ft_min:.1f} min) ==")
        stage_finetune(a, e, U, R, log, ft_min, bert_ok)
        log(f"== 4. {e['exp']} {e['name']}: baselines AE / VAE / ByteSGAN ==")
        try:
            stage_baselines(a, e, U, R, log, bl_sec)
        except Exception as ex:
            log(f"   !! baselines failed: {ex}"); log.problems.append(f"baselines {e['name']}")

    # 5-6. Evaluation, tables, figures, report
    log("== 5-6. Evaluation, tables and figures ==")
    res, figs, tables = stage_report(a, D, U, R, log, t_start)

    log("")
    log(f"{'experiment':<14}{'model':<14}{'Precision':>10}{'Recall':>9}{'F1':>9}{'ACC':>9}")
    for e in exps:
        for m in MODELS:
            r = res.get((e["key"], m))
            if r:
                log(f"{e['name']:<14}{m:<14}{r['macro']['precision']:>10.4f}{r['macro']['recall']:>9.4f}"
                    f"{r['macro']['f1']:>9.4f}{r['accuracy']:>9.4f}")
    log(f"\nDone in {(time.time() - t_start) / 60:.1f} min.  Report: {R}/report.md   "
        f"Figures: {R}/figures/ ({len(figs)})   Tables: {R}/tables/ ({len(tables)})")
    if log.problems:
        log("Some steps failed: " + ", ".join(log.problems) + "  (see pipeline.log)")


if __name__ == "__main__":
    main()
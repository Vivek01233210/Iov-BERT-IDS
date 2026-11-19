"""
Fine-tune mini IoV-BERT for intrusion detection on a USL file and evaluate it.

Examples
  # Exp 1/2 style: one dataset, 80/20 split
  python finetune.py --usl data/car_hacking_usl.tsv --bert runs/pretrain/bert.pt --out runs/carhack

  # Ablation: same thing without pre-training (does pre-training help?)
  python finetune.py --usl data/car_hacking_usl.tsv --no-pretrain --preset small --out runs/carhack_scratch

  # Exp 3 style: train on one vehicle, test on another
  python finetune.py --usl data/sonata_usl.tsv --test-usl data/soul_usl.tsv --bert runs/pretrain/bert.pt --out runs/cross

  # Leak-free split: whole sessions / CAN time-blocks go to either train or test
  python finetune.py --usl data/cicids_usl.tsv --group-split --bert runs/pretrain/bert.pt --out runs/cicids
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from data import batches, classify_batch, load_usl
from model import IoVBertClassifier, MiniBert, build_bert, count_params


def to_torch(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, sents, label_ids, idx_all, max_len, bs, device):
    model.eval()
    preds = []
    for s in range(0, len(idx_all), bs):
        idx = idx_all[s:s + bs]
        b = to_torch(classify_batch(sents, label_ids, idx, max_len), device)
        logits = model(b["input_ids"], b["segment_ids"], b["pad_mask"])
        preds.append(logits.argmax(-1).cpu().numpy())
    model.train()
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def report(y_true, y_pred, names, title):
    labels = list(range(len(names)))
    acc = accuracy_score(y_true, y_pred)
    out = {"accuracy": acc}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out[avg] = {"precision": p, "recall": r, "f1": f}
    print(f"\n===== {title} =====")
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, digits=4, zero_division=0))
    print(f"accuracy {acc:.4f} | macro P/R/F1 {out['macro']['precision']:.4f} / "
          f"{out['macro']['recall']:.4f} / {out['macro']['f1']:.4f}")
    out["per_class"] = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                             output_dict=True, zero_division=0)
    return out


def plot_cm(y_true, y_pred, names, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * n, 0.8 + 0.6 * n))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usl", required=True, help="training USL (label \\t group \\t sentence)")
    ap.add_argument("--test-usl", help="separate test set (e.g. another vehicle). Otherwise --test-size split")
    ap.add_argument("--bert", help="pre-trained bert.pt from pretrain.py")
    ap.add_argument("--no-pretrain", action="store_true", help="train from random init (ablation)")
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base"],
                    help="model size when --no-pretrain")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--out", default="runs/finetune")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1, help="fraction of train used for early stopping")
    ap.add_argument("--group-split", action="store_true", help="split by group column (no leakage)")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze embeddings + first N encoder layers")
    ap.add_argument("--class-weights", action="store_true", help="weight loss by inverse class frequency")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop training after this many minutes (then validate, test and save as usual)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    labels, groups, sents = load_usl(a.usl)
    if a.test_usl:
        t_labels, t_groups, t_sents = load_usl(a.test_usl)
    names = sorted(set(labels) | (set(t_labels) if a.test_usl else set()))
    lid = {n: i for i, n in enumerate(names)}
    y = np.array([lid[l] for l in labels])
    print(f"{len(sents)} training rows, classes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- splits ------------------------------------------------------------------------
    idx = np.arange(len(sents))
    if a.test_usl:
        train_idx = idx
        all_sents = sents + t_sents
        all_y = np.concatenate([y, [lid[l] for l in t_labels]])
        test_idx = np.arange(len(sents), len(all_sents))
    else:
        all_sents, all_y = sents, y
        if a.group_split:
            gss = GroupShuffleSplit(n_splits=1, test_size=a.test_size, random_state=a.seed)
            train_idx, test_idx = next(gss.split(idx, y, groups))
        else:
            train_idx, test_idx = train_test_split(idx, test_size=a.test_size, random_state=a.seed,
                                                   stratify=y if min(np.bincount(y)) >= 2 else None)
    tr_y = all_y[train_idx]
    strat = tr_y if min(np.bincount(tr_y, minlength=len(names))[np.unique(tr_y)]) >= 2 else None
    train_idx, val_idx = train_test_split(train_idx, test_size=a.val_size, random_state=a.seed, stratify=strat)
    print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")

    # ---- model -------------------------------------------------------------------------
    if a.no_pretrain or not a.bert:
        if not a.no_pretrain:
            print("WARNING: no --bert given, training from scratch")
        bert = build_bert(a.preset, max_len=a.max_len)
    else:
        ck = torch.load(a.bert, map_location="cpu", weights_only=False)
        bert = MiniBert(**ck["config"])
        bert.load_state_dict(ck["bert"])
        print(f"loaded pre-trained encoder {a.bert}")
    max_len = bert.config["max_len"] if a.max_len is None else min(a.max_len, bert.config["max_len"])
    model = IoVBertClassifier(bert, len(names)).to(device)
    if a.freeze_layers:
        for p in list(bert.tok.parameters()) + list(bert.seg.parameters()) + list(bert.pos.parameters()):
            p.requires_grad = False
        for layer in bert.encoder.layers[:a.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    print(f"params {count_params(model)/1e6:.1f}M, max_len {max_len}, device {device}")

    weight = None
    if a.class_weights:
        cnt = np.bincount(all_y[train_idx], minlength=len(names)).astype(np.float32)
        weight = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(names), device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    total = max(1, a.epochs * ((len(train_idx) + a.batch_size - 1) // a.batch_size))
    budget_s = a.max_minutes * 60 if a.max_minutes else None
    t_start = time.time()

    def progress(step):
        p = step / total
        if budget_s:
            p = max(p, (time.time() - t_start) / budget_s)
        return p

    def set_lr(step):
        """10% linear warm-up, then linear decay to 0 over the step or time budget."""
        p = progress(step)
        lr = a.lr * (p / 0.1 if p < 0.1 else max(0.0, (1 - p) / 0.9))
        for g in opt.param_groups:
            g["lr"] = max(lr, a.lr * 0.01)

    best_f1, best_path = -1.0, os.path.join(a.out, "best.pt")
    history = []
    step = 0
    out_of_time = False
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        run_loss, n = 0.0, 0
        for bidx in batches(len(train_idx), a.batch_size, True, rng):
            if progress(step) >= 1.0:
                out_of_time = True
                break
            set_lr(step)
            b = to_torch(classify_batch(all_sents, all_y, train_idx[bidx], max_len), device)
            logits = model(b["input_ids"], b["segment_ids"], b["pad_mask"])
            loss = F.cross_entropy(logits, b["labels"], weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            run_loss += loss.item(); n += 1; step += 1
            if step % 50 == 0:
                print(f"  epoch {ep} step {step}/{total} loss {run_loss / n:.4f} "
                      f"({(time.time() - t0) / n:.2f}s/step)", flush=True)
        vp = predict(model, all_sents, all_y, val_idx, max_len, a.batch_size * 2, device)
        _, _, vf1, _ = precision_recall_fscore_support(all_y[val_idx], vp, average="macro", zero_division=0)
        history.append({"epoch": ep, "train_loss": run_loss / max(n, 1), "val_macro_f1": vf1,
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {ep}: train loss {run_loss / max(n, 1):.4f}  val macro-F1 {vf1:.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save({"config": bert.config, "names": names, "model": model.state_dict()}, best_path)
        if out_of_time:
            print(f"time budget reached after {step} steps ({(time.time() - t_start) / 60:.1f} min)")
            break

    # ---- test --------------------------------------------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    t0 = time.time()
    tp = predict(model, all_sents, all_y, test_idx, max_len, a.batch_size * 2, device)
    ms_per_sample = (time.time() - t0) * 1000 / max(len(test_idx), 1)
    title = "TEST (" + (os.path.basename(a.test_usl) if a.test_usl else "held-out split") + ")"
    res = report(all_y[test_idx], tp, names, title)
    res["inference_ms_per_sample"] = ms_per_sample
    res["history"] = history
    res["args"] = vars(a)
    print(f"inference: {ms_per_sample:.2f} ms per sample on {device}")
    plot_cm(all_y[test_idx], tp, names, os.path.join(a.out, "confusion_matrix.png"), title)
    np.savez(os.path.join(a.out, "predictions.npz"), y_true=all_y[test_idx], y_pred=tp,
             names=np.array(names), test_idx=np.asarray(test_idx), train_idx=np.asarray(train_idx),
             val_idx=np.asarray(val_idx))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved results.json, confusion_matrix.png and best.pt in {a.out}")


if __name__ == "__main__":
    main()

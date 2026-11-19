"""
Pre-train mini IoV-BERT on a BSL file with MBWM + NBSP.

  python pretrain.py --bsl data/bsl.tsv --preset small --steps 20000 --out runs/pretrain

Runs on CPU by default and uses CUDA automatically if present (e.g. Kaggle / Colab).
Saves a checkpoint every --save-every steps; re-run with --resume to continue
after closing the laptop.
"""
import argparse
import json
import math
import os
import random
import time

import torch

from data import load_bsl, pretrain_batch
from model import IoVBertPretrain, build_bert, count_params


def to_torch(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def lr_at(step, progress, warmup, peak):
    """Linear warm-up (ends after `warmup` steps or 5% of the budget, whichever is first),
    then cosine decay over `progress` (0..1 of the step or time budget)."""
    warm = min(1.0, max((step + 1) / warmup, progress / 0.05))
    return peak * warm * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsl", required=True)
    ap.add_argument("--out", default="runs/pretrain")
    ap.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--max-len", type=int, default=None, help="override preset sequence length")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop after this many minutes; the learning-rate schedule adapts to the budget")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N pairs")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    pairs = load_bsl(a.bsl, a.limit)
    print(f"loaded {len(pairs)} sentence pairs from {a.bsl}")
    if len(pairs) < 2:
        raise SystemExit("need at least 2 pairs")

    bert = build_bert(a.preset, max_len=a.max_len)
    model = IoVBertPretrain(bert).to(device)
    max_len = bert.config["max_len"]
    print(f"preset={a.preset} config={bert.config} params={count_params(model)/1e6:.1f}M device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step = 0
    ckpt_path = os.path.join(a.out, "checkpoint.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        rng.setstate(ck["rng"])
        print(f"resumed from step {step}")

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "rng": rng.getstate()}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        # encoder-only weights + config for fine-tuning
        torch.save({"config": bert.config, "bert": bert.state_dict()}, os.path.join(a.out, "bert.pt"))

    model.train()
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = t_start = time.time()
    warmup = min(a.warmup, max(1, a.steps // 10))
    budget_s = a.max_minutes * 60 if a.max_minutes else None

    def progress():
        p = step / a.steps
        if budget_s:
            p = max(p, (time.time() - t_start) / budget_s)
        return p

    agg = dict(loss=0.0, mbwm=0.0, nbsp=0.0, mbwm_acc=0.0, nbsp_acc=0.0)
    n_agg = 0
    while progress() < 1.0:
        idx = [rng.randrange(len(pairs)) for _ in range(a.batch_size)]
        batch = to_torch(pretrain_batch(pairs, idx, max_len, rng), device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, progress(), warmup, a.lr)
        loss, l_m, l_n, acc_m, acc_n = model(**batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in zip(agg, (loss, l_m, l_n, acc_m, acc_n)):
            agg[k] += float(v)
        n_agg += 1
        if step % a.log_every == 0 or progress() >= 1.0:
            rec = {k: round(v / n_agg, 4) for k, v in agg.items()}
            rec.update(step=step, lr=round(opt.param_groups[0]["lr"], 7),
                       sec_per_step=round((time.time() - t0) / n_agg, 3))
            eta_h = rec["sec_per_step"] * (a.steps - step) / 3600
            if budget_s:
                eta_h = min(eta_h, max(0.0, budget_s - (time.time() - t_start)) / 3600)
            print(f"step {step:>6}/{a.steps}  loss {rec['loss']:.3f}  "
                  f"MBWM {rec['mbwm']:.3f} (acc {rec['mbwm_acc']:.2f})  "
                  f"NBSP {rec['nbsp']:.3f} (acc {rec['nbsp_acc']:.2f})  "
                  f"{rec['sec_per_step']:.2f}s/step  ETA {eta_h * 60:.1f} min", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            agg = {k: 0.0 for k in agg}; n_agg = 0; t0 = time.time()
        if step % a.save_every == 0:
            save()
    save()
    print(f"done after {step} steps ({(time.time() - t_start) / 60:.1f} min). encoder weights: {os.path.join(a.out, 'bert.pt')}")


if __name__ == "__main__":
    main()

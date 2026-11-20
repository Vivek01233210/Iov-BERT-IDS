# IoV-BERT-IDS Mini: a laptop version of IoV-BERT-IDS

This is a scaled-down implementation of *IoV-BERT-IDS: Hybrid Network Intrusion Detection System in IoV Using Large Language Models* (Fu et al., IEEE TVT 2025).

The pipeline is the same as in the paper:

```
raw traffic ──► Semantic Extractor ──► Byte Sentences ──► mini-BERT pre-training (MBWM + NBSP)
                (BSE / USE)            (BSL / USL)                 │
                                                                   ▼
                     CAN logs / labelled pcaps ──► USL ──► fine-tuning ──► intrusion detection
```

## What was scaled down, and why

| | Paper | This project (default `small`) |
|---|---|---|
| Model | BERT-base: 12 layers, 768 hidden, 12 heads, about 110M params | 4 layers, 256 hidden, 4 heads, about 20M params (`tiny`: 2 layers, 128 hidden, about 9M) |
| Sequence length | not stated | 128 tokens (`tiny`: 64) |
| Pre-training data | about 40 GB of PCAPs | about 0.3–2 GB of PCAPs (capped at 300k sentence pairs) |
| Fine-tuning data | up to 10k–20k samples per class | the same (10k per class), or fewer |
| Hardware | RTX 4080 16 GB | laptop CPU (a GPU is used automatically if one is present) |
| Vocabulary, SE, MBWM/NBSP, 80/10/10 masking, [CLS] classifier | ✔ | ✔ identical |

Most of the parameters sit in the 65,536-word byte-word embedding, which is cheap to compute. The transformer itself is small.

The MBWM head only computes vocabulary logits at the masked positions. Without that trick, a 65k-way softmax over every token would not fit comfortably in 8 GB of RAM.


## 1. Install (about 5 minutes)

You need Python 3.9–3.12.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/Mac:  source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install numpy scikit-learn matplotlib
```

No pcap libraries are needed. `pcap_reader.py` reads `.pcap` and `.pcapng` files itself.

## 2. Run the whole pipeline (about 10 minutes)

```bash
python run_pipeline.py --data datasets          # real datasets, laid out as below
python run_pipeline.py --data datasets          # or data made by generate_datasets.py (section 3b)
python run_pipeline.py --data datasets --minutes 30 --per-class 5000 --preset small   # bigger run, better scores
```

Expected layout of the data folder. Any dataset that is missing is skipped, along with its experiment:

```
<data>/cicids/       *.pcap + labelled-flow *.csv  (CICIDS2017 GeneratedLabelledFlows)
<data>/botiot/       <ClassName>/*.pcap
<data>/car_hacking/  DoS_dataset.csv  Fuzzy_dataset.csv  gear_dataset.csv  RPM_dataset.csv  normal_run_data.txt
<data>/ivn/          Sonata/*.csv  Soul/*.csv  Spark/*.csv
<data>/pretrain/     (optional) unlabelled pcaps for pre-training, e.g. your own Wireshark capture
```

The stages, in the paper's order:

| Stage | What happens | Share of the time budget |
|---|---|---|
| 1. Semantic Extractor | BSE → BSL for pre-training; USE → USL for each dataset; per-class sampling (`--per-class`, default 1,000; the paper used 10,000) | a few seconds on synthetic data |
| 2. Pre-training | MBWM + NBSP on the BSL (`tiny` model) | ~33% |
| 3. Fine-tuning | IoV-BERT-IDS on Exp 1 (CICIDS, BoT-IoT), Exp 2 (IVN-IDS, Car-Hacking) and Exp 3 (train Sonata → test Soul) | ~42% |
| 4. Baselines | AE, VAE and ByteSGAN on exactly the same train/test split | ~13% |
| 5–6. Evaluation and report | Precision, Recall, F1, Accuracy (Eqs. 11–14), confusion matrices, tables, figures | a few seconds |

Training stops on a clock (`--max-minutes` inside `pretrain.py` and `finetune.py`), so the run takes about the same time on any laptop. A slower laptop simply trains for fewer steps.

**Outputs in `results/`:**

| File | Matches the paper's |
|---|---|
| `tables/table1_cicids_dataset` … `table4_ivn_dataset` (.png, .csv, .md) | Tables I–IV (classes, original vs. used samples) |
| `tables/table5_environment` | Table V (equipment configuration) |
| `tables/table6_exp1`, `table7_exp2`, `table8_exp3` | Tables VI–VIII (P / R / F1 / ACC per model; Exp 3 per class) |
| `figures/fig3_mbwm_example.png`, `fig4_nbsp_example.png` | Figs. 3–4, drawn on a real sample with the pre-trained model's predictions |
| `figures/fig5_can_structure.png` | Fig. 5 (CAN frame fields, with a sample row and its CAN Byte Sentence) |
| `figures/fig6_confusion_matrices.png` | Fig. 6 (confusion matrices: 4 datasets × 4 models) |
| `figures/fig7_precision_radar.png` | Fig. 7 (radar charts of per-class precision, CICIDS and BoT-IoT) |
| `figures/fig8_prf_bars.png` | Fig. 8 (per-class P/R/F1 bars, IVN-IDS and Car-Hacking) |
| `figures/fig9_generalisation.png` | Fig. 9 (cross-vehicle P/R/F1), plus `fig9b_exp3_confusion_matrices.png` |
| `figures/fig_pretraining_curves.png`, `fig_finetuning_curves.png` | Extra: training curves |
| `report.md` | Every table and figure in one document |
| `pipeline.log` | The full log of the run |

**Notes on the comparison:**
- The paper's baselines used flow-level features. Here AE, VAE and ByteSGAN are re-implemented and trained on the first 64 raw bytes of each packet or CAN frame, on the same split as IoV-BERT-IDS. `report.md` says this too.
- If the data folder contains `README_SYNTHETIC.txt` (written by `generate_datasets.py`), every figure, table and the report are labelled "synthetic data".
- Re-run with `--reuse` to skip stages whose outputs already exist (for example, to redraw the figures only).

Figs. 1 and 2 (IoV architecture and framework) are conceptual diagrams rather than results, so cite them from the paper or draw them yourself.

## 3. Get data

You don't need everything. **Start with Car-Hacking**: it is small and easy to process.

| Purpose | Dataset | Where | Notes |
|---|---|---|---|
| Pre-training (unlabelled) | **Your own capture** | Wireshark on your laptop: browse, stream video, and use apps for 30–60 min, then *File → Save As* | 200 MB – 1 GB is plenty. `.pcapng` is fine. |
| | *or* CICIDS2017 `Monday-WorkingHours.pcap` | unb.ca/cic/datasets/ids-2017.html | Benign-only day, about 10 GB. No need to cut it: the extractor streams the file and stops at `--max-pairs`. |
| In-vehicle fine-tuning | **Car-Hacking** (HCRL) | hksecurity.net → HCRL Datasets → Car-Hacking | `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv`, `normal_run_data.txt` |
| In-vehicle + cross-vehicle (Exp 3) | **In-vehicle Network Intrusion Detection Challenge 2019** (preliminary round): Sonata / Soul / Spark | hksecurity.net → HCRL Datasets → datachallenge2019 | Has an R/T `Class` column; use `--format generic` |
| Extra-vehicle fine-tuning | CICIDS2017 PCAPs + **GeneratedLabelledFlows** CSVs | unb.ca/cic/datasets/ids-2017.html | The *labelled-flows* CSVs contain Source/Destination IP and Port, which the 5-tuple matching needs. The *MachineLearningCVE* CSVs don't. Start with Wednesday (DoS attacks). |
| | BoT-IoT (optional) | research.unsw.edu.au/projects/bot-iot-dataset | If you arrange pcaps as `folder/<ClassName>/*.pcap`, use the folder layout |

> **Important:** this is simulated data. It is fine for building, debugging and running your experiments, but results on it are **not** results on the real datasets. Write "synthetic data" next to every number you report from it, and replace them with real-data results if you can later download at least the CAN datasets.

## 4. Semantic Extractor: raw data → byte sentences

```bash
# Pre-training list (BSL): consecutive packets in a session become sentence pairs
python semantic_extractor.py pretrain path/to/pcaps/ --out data/bsl.tsv --max-pairs 300000

# In-vehicle (Car-Hacking): one CAN frame -> one CAN Byte Sentence (CBS)
python semantic_extractor.py can "carhack/*" --out data/carhack_usl.tsv --max-per-class 10000

# In-vehicle challenge data, one file per vehicle for Experiment 3.
# Check the CSV header first and adjust --id-col / --data-cols / --label-col if needed.
python semantic_extractor.py can "ivn/Sonata/*.csv" --format generic --out data/sonata_usl.tsv
python semantic_extractor.py can "ivn/Soul/*.csv"   --format generic --out data/soul_usl.tsv
#   'T' rows get the class name from the first word of the file name
#   (Flooding_..., Fuzzy_..., Malfunction_...). Rename files or use --attack-name if needed.

# Extra-vehicle, CICIDS style: label packets by matching 5-tuples against the flow CSV
python semantic_extractor.py traffic cicids/pcap_wed/ --flows-csv cicids/csv/Wednesday*.csv \
       --out data/cicids_usl.tsv --max-per-class 10000

# Extra-vehicle, folder style: data/<Label>/*.pcap
python semantic_extractor.py traffic botiot_by_class/ --out data/botiot_usl.tsv
```

Output formats are plain TSV that you can open in any editor:
- BSL: `byte_sentence_a <TAB> byte_sentence_b`
- USL: `label <TAB> group <TAB> byte_sentence`

A byte sentence looks like `4745 4554 5420 202f …`: overlapping 2-byte words, as in Fig. 2 of the paper.

## 5. Pre-train (the slow part)

```bash
python pretrain.py --bsl data/bsl.tsv --preset small --steps 20000 --batch-size 32 --out runs/pretrain
```

- The script prints seconds per step and an **ETA** within the first minute. If the ETA is too long, lower `--steps` or switch to `--preset tiny`.
- As a rough expectation on a 6-core Ryzen 5 CPU, `tiny` runs at a fraction of a second per step and `small` at about half a second to a second per step. That means `small` for 20k steps is an overnight job. Measure on your own machine.
- A checkpoint is saved every 1,000 steps. If you close the laptop, re-run the same command with `--resume` to continue.
- **What healthy training looks like:** MBWM accuracy rises steadily from about 0, and NBSP accuracy moves from 0.5 towards 0.9 or more.
- The output `runs/pretrain/bert.pt` holds the encoder weights you will fine-tune.

**Optional: use a free cloud GPU for this step only.** Kaggle Notebooks and Google Colab both offer a free GPU quota. Upload this folder and `bsl.tsv`, then run the same command; it uses CUDA automatically. There you can afford `--preset small --steps 50000` or even `--preset base`. Download `bert.pt` and do everything else on your laptop.

## 6. Fine-tune and evaluate

```bash
# Experiment 2 (in-vehicle, Car-Hacking)
python finetune.py --usl data/carhack_usl.tsv --bert runs/pretrain/bert.pt --epochs 3 --out runs/carhack

# Experiment 1 (extra-vehicle, CICIDS)
python finetune.py --usl data/cicids_usl.tsv --bert runs/pretrain/bert.pt --epochs 3 --out runs/cicids

# Experiment 3 (cross-vehicle generalisation: train Sonata, test Soul)
python finetune.py --usl data/sonata_usl.tsv --test-usl data/soul_usl.tsv --bert runs/pretrain/bert.pt --out runs/cross

# Ablation: does pre-training help? (same data, random initialisation)
python finetune.py --usl data/carhack_usl.tsv --no-pretrain --preset small --out runs/carhack_scratch

# Stricter evaluation: no session or time-block appears in both train and test
python finetune.py --usl data/cicids_usl.tsv --group-split --bert runs/pretrain/bert.pt --out runs/cicids_group
```

Each run writes the following to its `--out` folder:
- `results.json`: per-class precision/recall/F1, macro and weighted averages, accuracy, and inference ms per sample
- `confusion_matrix.png`
- `best.pt`: the best checkpoint by validation macro-F1

Useful flags:
- `--batch-size 32` if RAM is tight
- `--freeze-layers 2` for faster fine-tuning
- `--class-weights` for imbalanced classes
- `--max-len 64`: CAN sentences are short, so this speeds things up

## 7. Tips for a basic laptop

- Close the browser and other heavy apps while training, and keep the laptop plugged in with sleep disabled.
- If you run out of memory, lower `--batch-size`, use `--preset tiny`, or pass `--limit 100000` to `pretrain.py`. Car-Hacking also works fine with `--max-per-class 5000`.
- `--threads` defaults to all CPU cores. Set it to (cores − 1) if you want the laptop to stay usable.
- Fine-tuning is much cheaper than pre-training. A 50k-row CAN dataset usually takes minutes per epoch on `small`.

## 8. Writing it up

State clearly that this is a scaled-down reproduction: model size, pre-training corpus size and hardware differ, as in the table above. Report the ablation (with vs. without pre-training) and the group split alongside the standard split. Those two results say more about whether the idea works than near-100% accuracy on Car-Hacking, which simple models also reach.

## Files

| File | What it does |
|---|---|
| `pcap_reader.py` | Streaming `.pcap`/`.pcapng` reader (IPv4 TCP/UDP/ICMP) |
| `semantic_extractor.py` | BSE / USE → BSL / USL files; CAN extractor |
| `data.py` | Byte-word tokenizer, MBWM masking, NBSP pairs, batching (numpy) |
| `model.py` | Mini BERT encoder, pre-training heads, classifier |
| `pretrain.py` | MBWM + NBSP pre-training with checkpoints and resume |
| `finetune.py` | Fine-tuning, evaluation, confusion matrix, ablations |
| `run_pipeline.py` | The whole paper pipeline: extraction, pre-training, fine-tuning, baselines, Tables I–VIII, Figs. 3–9, report |
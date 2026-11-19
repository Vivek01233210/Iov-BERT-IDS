"""
Tokenizer + batch building (pure numpy, no torch needed here).

Vocabulary: every 2-byte byte word 0000..ffff (65,536) + 5 special tokens.
Because every byte word can be enumerated, there is no out-of-vocabulary problem;
[UNK] only catches malformed input.
"""
import numpy as np

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
N_SPECIAL = 5
VOCAB_SIZE = 65536 + N_SPECIAL
IGNORE = -100


def encode(sentence):
    """'facb cb34' -> [id, id] ."""
    ids = []
    for w in sentence.split():
        try:
            ids.append(int(w, 16) + N_SPECIAL if len(w) == 4 else UNK)
        except ValueError:
            ids.append(UNK)
    return ids


# --------------------------------------------------------------------------------------
# File loaders
# --------------------------------------------------------------------------------------
def load_bsl(path, limit=None):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                pairs.append((parts[0], parts[1]))
                if limit and len(pairs) >= limit:
                    break
    return pairs


def load_usl(path):
    labels, groups, sents = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                lab, g, s = parts
            elif len(parts) == 2:           # label \t sentence
                lab, s = parts
                g = str(len(sents))
            else:
                continue
            if s:
                labels.append(lab)
                groups.append(g)
                sents.append(s)
    return labels, groups, sents


# --------------------------------------------------------------------------------------
# Pre-training batches: MBWM + NBSP
# --------------------------------------------------------------------------------------
def make_pair_example(a_ids, b_ids, max_len):
    """[CLS] a [SEP] b [SEP]  with segment ids 0..0 1..1"""
    budget = max_len - 3
    while len(a_ids) + len(b_ids) > budget:          # trim the longer one
        if len(a_ids) >= len(b_ids):
            a_ids = a_ids[:-1]
        else:
            b_ids = b_ids[:-1]
    ids = [CLS] + a_ids + [SEP] + b_ids + [SEP]
    seg = [0] * (len(a_ids) + 2) + [1] * (len(b_ids) + 1)
    return ids, seg


def mask_tokens(ids, rng, mask_prob=0.15):
    """BERT 80/10/10 masking. Returns (masked_ids, labels)."""
    ids = list(ids)
    labels = [IGNORE] * len(ids)
    cand = [i for i, t in enumerate(ids) if t >= N_SPECIAL]
    if not cand:
        return ids, labels
    n = max(1, int(round(len(cand) * mask_prob)))
    for i in rng.sample(cand, min(n, len(cand))):
        labels[i] = ids[i]
        r = rng.random()
        if r < 0.8:
            ids[i] = MASK
        elif r < 0.9:
            ids[i] = rng.randrange(N_SPECIAL, VOCAB_SIZE)
        # else keep the original token
    return ids, labels


def pad_batch(seqs, value):
    L = max(len(s) for s in seqs)
    out = np.full((len(seqs), L), value, dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def pretrain_batch(pairs, idx, max_len, rng, mask_prob=0.15):
    """pairs: list of (sent_a, sent_b). idx: indices for this batch.
    50% of the time sentence_b is swapped for a random one (NotNext = 1)."""
    ids_l, seg_l, lab_l, nsp_l = [], [], [], []
    for i in idx:
        a, b = pairs[i]
        is_next = rng.random() < 0.5
        if not is_next:
            j = rng.randrange(len(pairs))
            if j == i:
                j = (j + 1) % len(pairs)
            b = pairs[j][1]
        ids, seg = make_pair_example(encode(a), encode(b), max_len)
        ids, lab = mask_tokens(ids, rng, mask_prob)
        ids_l.append(ids); seg_l.append(seg); lab_l.append(lab)
        nsp_l.append(0 if is_next else 1)
    ids = pad_batch(ids_l, PAD)
    return {
        "input_ids": ids,
        "segment_ids": pad_batch(seg_l, 0),
        "pad_mask": ids == PAD,
        "mlm_labels": pad_batch(lab_l, IGNORE),
        "nsp_labels": np.array(nsp_l, dtype=np.int64),
    }


# --------------------------------------------------------------------------------------
# Fine-tuning batches: [CLS] sentence [SEP]
# --------------------------------------------------------------------------------------
def classify_batch(sents, label_ids, idx, max_len):
    seqs = []
    for i in idx:
        ids = encode(sents[i])[: max_len - 2]
        seqs.append([CLS] + ids + [SEP])
    ids = pad_batch(seqs, PAD)
    return {
        "input_ids": ids,
        "segment_ids": np.zeros_like(ids),
        "pad_mask": ids == PAD,
        "labels": np.array([label_ids[i] for i in idx], dtype=np.int64),
    }


def batches(n, batch_size, shuffle, rng):
    order = list(range(n))
    if shuffle:
        rng.shuffle(order)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]

"""
Mini IoV-BERT in plain PyTorch (no HuggingFace needed).

Presets (paper = BERT-base: 12 layers, 768 hidden, 12 heads, ~110M params):
  tiny  : 2 layers, 128 hidden, 2 heads  ->  ~8.8M params
  small : 4 layers, 256 hidden, 4 heads  -> ~20M params
  base  : 12 layers, 768 hidden, 12 heads -> the paper's size (needs a GPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE, IGNORE

PRESETS = {
    "tiny":  dict(layers=2,  hidden=128, heads=2,  ffn=512,  max_len=64),
    "small": dict(layers=4,  hidden=256, heads=4,  ffn=1024, max_len=128),
    "base":  dict(layers=12, hidden=768, heads=12, ffn=3072, max_len=128),
}


class MiniBert(nn.Module):
    def __init__(self, layers, hidden, heads, ffn, max_len, dropout=0.1, activation="gelu"):
        super().__init__()
        self.config = dict(layers=layers, hidden=hidden, heads=heads, ffn=ffn,
                           max_len=max_len, dropout=dropout, activation=activation)
        # Input embedding = token + segment + position  (paper Sec. IV-C)
        self.tok = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=0)
        self.seg = nn.Embedding(2, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.emb_norm = nn.LayerNorm(hidden)
        self.emb_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, ffn, dropout, activation=activation,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.apply(self._init)
        with torch.no_grad():
            self.tok.weight[0].zero_()          # [PAD] embedding

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, input_ids, segment_ids, pad_mask):
        L = input_ids.size(1)
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok(input_ids) + self.seg(segment_ids) + self.pos(pos)
        x = self.emb_drop(self.emb_norm(x))
        return self.encoder(x, src_key_padding_mask=pad_mask)      # (B, L, H)


class Pooler(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, h):
        return torch.tanh(self.dense(h[:, 0]))                      # [CLS]


class IoVBertPretrain(nn.Module):
    """MiniBert + Masked Byte Word Model head + Next Byte Sentence Prediction head."""

    def __init__(self, bert: MiniBert):
        super().__init__()
        H = bert.config["hidden"]
        self.bert = bert
        self.mlm_transform = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H))
        self.mlm_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))
        self.pooler = Pooler(H)
        self.nsp = nn.Linear(H, 2)

    def forward(self, input_ids, segment_ids, pad_mask, mlm_labels, nsp_labels):
        h = self.bert(input_ids, segment_ids, pad_mask)
        # MBWM: only compute 65k-way logits at masked positions (saves lots of CPU/RAM)
        sel = mlm_labels != IGNORE
        nsp_logits = self.nsp(self.pooler(h))
        loss_nbsp = F.cross_entropy(nsp_logits, nsp_labels)
        if sel.any():
            hm = self.mlm_transform(h[sel])                          # (n_masked, H)
            logits = hm @ self.bert.tok.weight.t() + self.mlm_bias  # tied weights
            target = mlm_labels[sel]
            loss_mbwm = F.cross_entropy(logits, target)
            mbwm_acc = (logits.argmax(-1) == target).float().mean().detach()
        else:
            loss_mbwm = h.sum() * 0.0
            mbwm_acc = torch.tensor(0.0)
        nbsp_acc = (nsp_logits.argmax(-1) == nsp_labels).float().mean().detach()
        # L = L_MBWM + L_NBSP   (paper Eq. 10)
        return loss_mbwm + loss_nbsp, loss_mbwm, loss_nbsp, mbwm_acc, nbsp_acc


class IoVBertClassifier(nn.Module):
    """Fine-tuning: [CLS] -> pooler -> dropout -> linear -> softmax (in the loss)."""

    def __init__(self, bert: MiniBert, num_labels):
        super().__init__()
        H = bert.config["hidden"]
        self.bert = bert
        self.pooler = Pooler(H)
        self.drop = nn.Dropout(bert.config["dropout"])
        self.out = nn.Linear(H, num_labels)

    def forward(self, input_ids, segment_ids, pad_mask):
        h = self.bert(input_ids, segment_ids, pad_mask)
        return self.out(self.drop(self.pooler(h)))


def build_bert(preset="small", **overrides):
    cfg = dict(PRESETS[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return MiniBert(**cfg)


def count_params(m):
    return sum(p.numel() for p in m.parameters())

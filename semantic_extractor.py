"""
Semantic Extractor (SE) from IoV-BERT-IDS, laptop edition.

Turns raw traffic into "Byte Sentences" that a BERT model can read.

  Unidirectional SE (USE): hex string -> byte units (2 hex chars) -> overlapping
      2-byte "byte words" (bigrams), e.g.  fa cb 34 ae  ->  "facb cb34 34ae"
  Bidirectional SE (BSE): consecutive packets of the same session become
      (byte sentence_a, byte sentence_b) pairs for pre-training (sliding window 2, step 1).

Sub-commands (all stream their input, so big files are fine on 8 GB RAM):

  pretrain  pcaps (unlabelled)                    -> BSL  (bs_a \t bs_b)
  traffic   labelled pcaps (folder-per-class, or
            CICIDS-style flow CSV)                -> USL  (label \t group \t bs_a)
  can       Car-Hacking / generic CAN CSV         -> USL  (label \t group \t bs_a)

Run  `python semantic_extractor.py <sub-command> -h`  for options.
"""
import argparse
import csv
import glob
import os
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict

from pcap_reader import iter_packets


# --------------------------------------------------------------------------------------
# Unidirectional SE
# --------------------------------------------------------------------------------------
def use(hex_str, max_words=64):
    """Hex string -> byte sentence (space-separated overlapping 2-byte words)."""
    hex_str = hex_str.lower()
    if len(hex_str) % 2:
        hex_str = hex_str[:-1]
    if len(hex_str) < 2:
        return ""
    if len(hex_str) == 2:          # a single byte: pad to one word
        return hex_str + "00"
    words = [hex_str[i:i + 4] for i in range(0, len(hex_str) - 2, 2)]
    return " ".join(words[:max_words])


def packet_hex(pkt, mode="payload", skip_bytes=0, max_bytes=65):
    """
    Bytes of a packet that the model will see, as hex.
      payload : transport payload only (what the paper does)
      packet  : IP header + transport header + payload, with IP addresses and ports
                zeroed so the model cannot just memorise hosts (lets header-only
                attacks such as scans/SYN floods, which have empty payloads, be seen)
    max_bytes=65 gives 64 byte words.
    """
    if mode == "payload":
        data = pkt["payload"][skip_bytes:]
    else:
        ip = bytearray(pkt["ip_header"])
        if len(ip) >= 20:
            ip[12:20] = b"\x00" * 8          # src/dst IP
            ip[10:12] = b"\x00\x00"          # checksum
            ip[4:6] = b"\x00\x00"            # IP ID
        l4 = bytearray(pkt["l4_header"])
        if pkt["proto"] in (6, 17) and len(l4) >= 4:
            l4[0:4] = b"\x00" * 4            # ports
        data = bytes(ip) + bytes(l4) + pkt["payload"]
    return data[:max_bytes].hex()


def session_key(pkt):
    """Direction-independent 5-tuple = a session (bi-directional flow)."""
    a = (pkt["src"], pkt["sport"])
    b = (pkt["dst"], pkt["dport"])
    lo, hi = (a, b) if a <= b else (b, a)
    return (lo[0], lo[1], hi[0], hi[1], pkt["proto"])


def flow_key(src, sport, dst, dport, proto):
    a, b = (src, int(sport)), (dst, int(dport))
    lo, hi = (a, b) if a <= b else (b, a)
    return (lo[0], lo[1], hi[0], hi[1], int(proto))


def list_pcaps(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.pcap", "*.pcapng", "*.cap"):
                out += glob.glob(os.path.join(p, "**", ext), recursive=True)
        else:
            out += glob.glob(p)
    return sorted(set(out))


# --------------------------------------------------------------------------------------
# Bidirectional SE  ->  BSL (pre-training data)
# --------------------------------------------------------------------------------------
def cmd_pretrain(a):
    pcaps = list_pcaps(a.inputs)
    if not pcaps:
        sys.exit("no pcap files found")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_pairs = 0
    # session -> [last_sentence, packets_seen, pairs_emitted]; LRU-bounded to cap RAM
    sessions = OrderedDict()

    def flush_single(state):
        nonlocal n_pairs
        sent, seen, _ = state
        if seen == 1:                               # single-packet session: split in half
            words = sent.split()
            if len(words) >= 2:
                mid = len(words) // 2
                out.write(" ".join(words[:mid]) + "\t" + " ".join(words[mid:]) + "\n")
                n_pairs += 1

    with open(a.out, "w") as out:
        for path in pcaps:
            print(f"[pretrain] reading {path}", flush=True)
            for pkt in iter_packets(path):
                hx = packet_hex(pkt, a.mode, a.skip_bytes, a.max_words + 1)
                sent = use(hx, a.max_words)
                if len(sent) < 4 * a.min_words:
                    continue
                k = session_key(pkt)
                st = sessions.get(k)
                if st is None:
                    sessions[k] = [sent, 1, 0]
                    if len(sessions) > a.max_open_sessions:
                        _, old = sessions.popitem(last=False)
                        flush_single(old)
                else:
                    sessions.move_to_end(k)
                    if st[2] < a.max_pairs_per_session:
                        out.write(st[0] + "\t" + sent + "\n")
                        n_pairs += 1
                        st[2] += 1
                    st[0] = sent
                    st[1] += 1
                if n_pairs >= a.max_pairs:
                    break
            if n_pairs >= a.max_pairs:
                break
        for st in sessions.values():
            if n_pairs >= a.max_pairs:
                break
            flush_single(st)
    print(f"[pretrain] wrote {n_pairs} sentence pairs to {a.out}")


# --------------------------------------------------------------------------------------
# Reservoir sampler: keep at most k random examples per class in one pass
# --------------------------------------------------------------------------------------
class PerClassReservoir:
    def __init__(self, k, seed=0):
        self.k = k
        self.rng = random.Random(seed)
        self.items = defaultdict(list)
        self.seen = Counter()

    def add(self, label, item):
        self.seen[label] += 1
        bucket = self.items[label]
        if len(bucket) < self.k:
            bucket.append(item)
        else:
            j = self.rng.randrange(self.seen[label])
            if j < self.k:
                bucket[j] = item

    def write(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rows = [(lab, g, s) for lab, b in self.items.items() for (g, s) in b]
        self.rng.shuffle(rows)
        with open(path, "w") as f:
            for lab, g, s in rows:
                f.write(f"{lab}\t{g}\t{s}\n")
        import json
        with open(path + ".counts.json", "w") as cf:
            json.dump({lab: {"used": len(self.items[lab]), "original": self.seen[lab]}
                       for lab in sorted(self.items)}, cf, indent=1)
        print("class counts (kept / seen):")
        for lab in sorted(self.items):
            print(f"  {lab:<28} {len(self.items[lab]):>8} / {self.seen[lab]}")
        print(f"wrote {len(rows)} rows to {path}")


# --------------------------------------------------------------------------------------
# Labelled extra-vehicle traffic -> USL
# --------------------------------------------------------------------------------------
def load_flow_labels(csv_paths, benign="BENIGN"):
    """CICIDS-style CSVs (Source IP, Source Port, Destination IP, Destination Port,
    Protocol, Label) -> {session_key: label}. Attack labels win over benign."""
    labels = {}
    for p in csv_paths:
        print(f"[traffic] loading labels from {p}", flush=True)
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f)
            header = [h.strip() for h in next(r)]
            idx = {h: i for i, h in enumerate(header)}
            need = ["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Label"]
            miss = [n for n in need if n not in idx]
            if miss:
                sys.exit(f"{p}: missing columns {miss}")
            for row in r:
                try:
                    k = flow_key(row[idx["Source IP"]].strip(), row[idx["Source Port"]],
                                 row[idx["Destination IP"]].strip(), row[idx["Destination Port"]],
                                 row[idx["Protocol"]])
                except (ValueError, IndexError):
                    continue
                lab = row[idx["Label"]].strip().replace("�", "-")
                if k not in labels or labels[k] == benign:
                    labels[k] = lab
    print(f"[traffic] {len(labels)} labelled sessions")
    return labels


def cmd_traffic(a):
    res = PerClassReservoir(a.max_per_class, a.seed)
    flow_labels = load_flow_labels(a.flows_csv) if a.flows_csv else None
    jobs = []
    if flow_labels is None:
        # folder-per-class layout: <root>/<label>/*.pcap
        for d in sorted(os.listdir(a.root)):
            full = os.path.join(a.root, d)
            if os.path.isdir(full):
                jobs += [(p, d) for p in list_pcaps([full])]
    else:
        jobs = [(p, None) for p in list_pcaps([a.root])]
    if not jobs:
        sys.exit("no pcap files found")
    for path, folder_label in jobs:
        print(f"[traffic] reading {path}", flush=True)
        per_session = Counter()
        for pkt in iter_packets(path):
            k = session_key(pkt)
            lab = folder_label if folder_label else flow_labels.get(k)
            if lab is None:
                continue
            if per_session[k] >= a.max_packets_per_session:
                continue
            sent = use(packet_hex(pkt, a.mode, a.skip_bytes, a.max_words + 1), a.max_words)
            if len(sent) < 4 * a.min_words:
                continue
            per_session[k] += 1
            group = "%s|%d|%s|%d|%d" % k
            res.add(lab, (group, sent))
    res.write(a.out)


# --------------------------------------------------------------------------------------
# In-vehicle CAN -> USL
# --------------------------------------------------------------------------------------
_HCRL_TXT = re.compile(r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\S+\s+DLC:\s*(\d)\s*((?:[0-9a-fA-F]{2}\s*)*)")


def iter_car_hacking(path, attack_name):
    """Car-Hacking dataset (HCRL). Two formats:
       attack CSVs : ts,CAN_ID,DLC,DATA0..DATA{DLC-1},Flag   (Flag T = injected, R = normal)
       normal txt  : 'Timestamp: ... ID: 0350 000 DLC: 8 05 28 84 ...'
    Yields (can_id_hex, [data bytes hex], label)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Timestamp"):
                m = _HCRL_TXT.match(line)
                if m:
                    yield m.group(2), m.group(4).split(), "Normal"
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                dlc = int(parts[2])
            except ValueError:
                continue                         # header row
            data = parts[3:3 + dlc]
            flag = parts[3 + dlc].strip() if len(parts) > 3 + dlc else "R"
            yield parts[1], data, (attack_name if flag == "T" else "Normal")


def iter_generic_can(path, id_col, data_cols, label_col, delimiter=","):
    """Any CSV with a header. data_cols: one column holding space-separated bytes,
    or several columns (comma list), one byte each."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter=delimiter)
        for row in r:
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if len(data_cols) == 1:
                data = row.get(data_cols[0], "").split()
            else:
                data = [row.get(c, "") for c in data_cols]
            yield row.get(id_col, ""), [d for d in data if d], row.get(label_col, "")


def can_sentence(frames, max_words):
    """frames: list of (can_id, data). CAN ID (padded to 4 hex) + DATA[0-7] per frame."""
    hx = ""
    for cid, data in frames:
        cid = cid.strip().lower()
        if cid.startswith("0x"):
            cid = cid[2:]
        cid = cid.zfill(4) if len(cid) <= 4 else cid.zfill(8)
        hx += cid + "".join(d.lower().zfill(2) for d in data)
    return use(hx, max_words)


def cmd_can(a):
    res = PerClassReservoir(a.max_per_class, a.seed)
    files = []
    for p in a.inputs:
        files += sorted(glob.glob(p))
    if not files:
        sys.exit("no input files found")
    data_cols = [c.strip() for c in a.data_cols.split(",")] if a.data_cols else None
    for path in files:
        base = os.path.basename(path)
        attack = a.attack_name or re.split(r"[_.\- ]", base)[0]   # e.g. DoS_dataset.csv -> DoS
        attack = {"gear": "Gear", "rpm": "RPM", "fuzzy": "Fuzzy", "dos": "DoS",
                  "flooding": "Flooding", "malfunction": "Malfunction"}.get(attack.lower(), attack)
        print(f"[can] reading {path}  (attack name for 'T' rows: {attack})", flush=True)
        if a.format == "car-hacking":
            it = iter_car_hacking(path, attack)
        else:
            it = iter_generic_can(path, a.id_col, data_cols, a.label_col, a.delimiter)
        window = []
        for i, (cid, data, lab) in enumerate(it):
            if not cid or not lab:
                continue
            if lab.upper() in ("T", "ATTACK", "1"):          # flag-style labels (R/T)
                lab = attack
            elif lab.upper() in ("R", "NORMAL", "0", "ATTACK_FREE", "ATTACK-FREE"):
                lab = "Normal"
            window.append((cid, data, lab))
            if len(window) > a.window:
                window.pop(0)
            if len(window) < a.window:
                continue
            labs = [w[2] for w in window]
            attack_labs = [l for l in labs if l != "Normal"]
            label = attack_labs[-1] if attack_labs else "Normal"
            sent = can_sentence([(w[0], w[1]) for w in window], a.max_words)
            if not sent:
                continue
            group = f"{base}:{i // a.block}"          # contiguous time blocks, for leak-free splits
            res.add(label, (group, sent))
    res.write(a.out)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-words", type=int, default=64, help="byte words per sentence (default 64)")
    common.add_argument("--seed", type=int, default=0)

    traffic_common = argparse.ArgumentParser(add_help=False)
    traffic_common.add_argument("--mode", choices=["payload", "packet"], default="payload",
                                help="payload = paper; packet = headers (anonymised) + payload")
    traffic_common.add_argument("--skip-bytes", type=int, default=0,
                                help="drop first N payload bytes (paper uses 4)")
    traffic_common.add_argument("--min-words", type=int, default=2)

    p = sub.add_parser("pretrain", parents=[common, traffic_common], help="unlabelled pcaps -> BSL")
    p.add_argument("inputs", nargs="+", help="pcap files / folders / globs")
    p.add_argument("--out", required=True)
    p.add_argument("--max-pairs", type=int, default=300_000)
    p.add_argument("--max-pairs-per-session", type=int, default=20)
    p.add_argument("--max-open-sessions", type=int, default=200_000)
    p.set_defaults(fn=cmd_pretrain)

    t = sub.add_parser("traffic", parents=[common, traffic_common], help="labelled pcaps -> USL")
    t.add_argument("root", help="folder with <label>/*.pcap, or folder of pcaps when --flows-csv is used")
    t.add_argument("--flows-csv", nargs="*", help="CICIDS/CICFlowMeter CSV(s) used to label sessions")
    t.add_argument("--out", required=True)
    t.add_argument("--max-per-class", type=int, default=10_000)
    t.add_argument("--max-packets-per-session", type=int, default=5)
    t.set_defaults(fn=cmd_traffic)

    c = sub.add_parser("can", parents=[common], help="CAN logs -> USL")
    c.add_argument("inputs", nargs="+", help="files or globs")
    c.add_argument("--format", choices=["car-hacking", "generic"], default="car-hacking")
    c.add_argument("--id-col", default="Arbitration_ID")
    c.add_argument("--data-cols", default="Data", help="one column with space-separated bytes, or a comma list")
    c.add_argument("--label-col", default="Class")
    c.add_argument("--delimiter", default=",")
    c.add_argument("--attack-name", help="class name for 'T' (injected) rows; default = first word of file name")
    c.add_argument("--window", type=int, default=1, help="consecutive frames per sentence (paper = 1)")
    c.add_argument("--block", type=int, default=5000, help="frames per group block for leak-free splitting")
    c.add_argument("--out", required=True)
    c.add_argument("--max-per-class", type=int, default=10_000)
    c.set_defaults(fn=cmd_can)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

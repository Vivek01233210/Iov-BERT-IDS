"""
Minimal, dependency-free reader for .pcap and .pcapng files.

Streams packets one at a time, so a multi-GB capture never has to fit in RAM.
Yields parsed IPv4 TCP/UDP/ICMP packets as simple dicts.

Supported link types: Ethernet (1), raw IP (101 / 228), Linux cooked SLL (113).
Anything else (IPv6, ARP, ...) is silently skipped -- that is fine for this project.
"""
import struct

LINK_ETHERNET = 1
LINK_RAW = 101
LINK_RAW_IPV4 = 228
LINK_SLL = 113

PROTO_ICMP, PROTO_TCP, PROTO_UDP = 1, 6, 17


def _ip_from_frame(frame, linktype):
    """Return the IPv4 packet bytes inside a link-layer frame, or None."""
    if linktype == LINK_ETHERNET:
        if len(frame) < 14:
            return None
        eth_type = struct.unpack("!H", frame[12:14])[0]
        off = 14
        while eth_type == 0x8100 and len(frame) >= off + 4:  # VLAN tag(s)
            eth_type = struct.unpack("!H", frame[off + 2:off + 4])[0]
            off += 4
        return frame[off:] if eth_type == 0x0800 else None
    if linktype in (LINK_RAW, LINK_RAW_IPV4):
        return frame if frame and (frame[0] >> 4) == 4 else None
    if linktype == LINK_SLL:
        if len(frame) < 16:
            return None
        return frame[16:] if struct.unpack("!H", frame[14:16])[0] == 0x0800 else None
    return None


def parse_ipv4(ip):
    """Parse an IPv4 packet. Returns dict or None."""
    if len(ip) < 20 or (ip[0] >> 4) != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    total_len = struct.unpack("!H", ip[2:4])[0]
    ip = ip[:total_len] if total_len >= ihl else ip
    proto = ip[9]
    src = ".".join(str(b) for b in ip[12:16])
    dst = ".".join(str(b) for b in ip[16:20])
    l4 = ip[ihl:]
    sport = dport = 0
    payload = b""
    l4_header = b""
    if proto == PROTO_TCP and len(l4) >= 20:
        sport, dport = struct.unpack("!HH", l4[:4])
        thl = (l4[12] >> 4) * 4
        l4_header, payload = l4[:thl], l4[thl:]
    elif proto == PROTO_UDP and len(l4) >= 8:
        sport, dport = struct.unpack("!HH", l4[:4])
        l4_header, payload = l4[:8], l4[8:]
    elif proto == PROTO_ICMP and len(l4) >= 8:
        l4_header, payload = l4[:8], l4[8:]
    else:
        return None
    return {
        "src": src, "dst": dst, "sport": sport, "dport": dport, "proto": proto,
        "ip_header": ip[:ihl], "l4_header": l4_header, "payload": payload,
    }


def _iter_pcap(f, magic):
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    else:
        endian = ">"
    hdr = f.read(20)
    linktype = struct.unpack(endian + "HHiIII", hdr)[5] & 0x0FFFFFFF
    ts_div = 1e9 if magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d") else 1e6
    while True:
        rec = f.read(16)
        if len(rec) < 16:
            return
        ts_sec, ts_frac, incl_len, _ = struct.unpack(endian + "IIII", rec)
        data = f.read(incl_len)
        if len(data) < incl_len:
            return
        yield ts_sec + ts_frac / ts_div, linktype, data


def _iter_pcapng(f):
    linktypes = []
    endian = "<"
    while True:
        head = f.read(8)
        if len(head) < 8:
            return
        btype_raw, blen_raw = head[:4], head[4:]
        if btype_raw == b"\x0a\x0d\x0d\x0a":  # Section Header Block: detect endianness
            bom = f.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            blen = struct.unpack(endian + "I", blen_raw)[0]
            f.read(blen - 12)
            linktypes = []
            continue
        btype = struct.unpack(endian + "I", btype_raw)[0]
        blen = struct.unpack(endian + "I", blen_raw)[0]
        if blen < 12:
            return
        body = f.read(blen - 8)
        if len(body) < blen - 8:
            return
        if btype == 1:  # Interface Description Block
            linktypes.append(struct.unpack(endian + "H", body[:2])[0])
        elif btype == 6:  # Enhanced Packet Block
            iface, ts_hi, ts_lo, cap_len, _ = struct.unpack(endian + "IIIII", body[:20])
            lt = linktypes[iface] if iface < len(linktypes) else LINK_ETHERNET
            yield ((ts_hi << 32) | ts_lo) / 1e6, lt, body[20:20 + cap_len]
        elif btype == 3:  # Simple Packet Block
            cap_len = struct.unpack(endian + "I", body[:4])[0]
            lt = linktypes[0] if linktypes else LINK_ETHERNET
            yield 0.0, lt, body[4:4 + cap_len]


def iter_packets(path):
    """Yield parsed IPv4 packets (dicts with 'ts' added) from a .pcap or .pcapng file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            f.seek(0)
            raw_iter = _iter_pcapng(f)
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
            raw_iter = _iter_pcap(f, magic)
        else:
            raise ValueError(f"{path}: not a pcap/pcapng file")
        for ts, linktype, frame in raw_iter:
            ip = _ip_from_frame(frame, linktype)
            if ip is None:
                continue
            pkt = parse_ipv4(ip)
            if pkt is not None:
                pkt["ts"] = ts
                yield pkt

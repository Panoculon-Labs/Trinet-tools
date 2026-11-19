#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Panoculon Labs. Part of the Trinet toolkit.
"""Decrypt encrypted Trinet SD-card recordings.

A Trinet camera with an SD-encryption key installed encrypts each take (or
each part of a split recording) to the owner's RSA-2048 public key; only the
matching private key, which never leaves the owner, can decrypt it. See
https://docs.panoculonlabs.com/data/encryption/ for setting it up.

Command line (``python scripts/decrypt_recordings.py`` is the same tool):
  # once: make a key pair (keep priv.pem secret, back it up)
  python -m trinet_tools.decrypt keygen priv.pem pub.tvk [--passphrase]
  # pub.tvk is the camera's key: install it from the Trinet app
  python -m trinet_tools.decrypt export-pubkey priv.pem pub.tvk

  # decrypt a card / folder / single take into OUT (originals untouched)
  python -m trinet_tools.decrypt decrypt --key priv.pem /media/SD/Trinet --out ./plain

Output MP4s keep every box offset of the original (each decrypted sample is
padded back to its encrypted size with a filler NAL), and the .imu/.vts/.tel
sidecars come out byte-identical to an unencrypted recording, so the rest of
Trinet-tools (reader, metadata, ingest) works unchanged on the result.
"""
import argparse
import getpass
import hashlib
import os
import shutil
import struct
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEY_SEI_UUID = b"TRINETSDENCKEYv1"
TKEY_LEN = 272
SIDECARS = ("imu", "vts", "tel", "wav")
# Stereo (0.5.16): both eyes share one fileset key (<base>.tkey, one CK) but
# never a keystream. The _L file is lane 0 and uses the mono names; the _R
# file is lane 1: "video_R", "audio_R", and its .vts is "vts_R".
STEREO_EXTRA = ("video_R", "audio_R", "vts_R")
EYE_LANE = {"_L": 0, "_R": 1}


# ---------------------------------------------------------------- keys

def pubkey_to_tvk(pub):
    nums = pub.public_numbers()
    if pub.key_size != 2048:
        raise ValueError("camera format needs an RSA-2048 key")
    return b"TVK1" + struct.pack(">I", nums.e) + nums.n.to_bytes(256, "big")


def key_id_of(pub):
    return hashlib.sha256(pubkey_to_tvk(pub)[4:]).digest()[:8]


def load_private(path, passphrase=None):
    data = open(path, "rb").read()
    try:
        return serialization.load_pem_private_key(data, password=passphrase)
    except TypeError:
        pw = getpass.getpass(f"passphrase for {path}: ").encode()
        return serialization.load_pem_private_key(data, password=pw)


ALG_AES_CTR, ALG_CHACHA20 = 1, 2     # alg 1: first test builds only


class FileKeys:
    """Per-fileset stream keys from the RSA-OAEP wrapped content key.

    Each stream cipher call takes a per-message nonce built by the *_nonce
    helpers (laid out exactly as the camera does) and a byte offset into
    that keystream (sidecars are keyed by file offset)."""

    def __init__(self, priv, key_id, wrapped, alg=ALG_CHACHA20):
        if alg not in (ALG_AES_CTR, ALG_CHACHA20):
            raise ValueError(f"unknown cipher alg {alg}")
        self.alg = alg
        if key_id != key_id_of(priv.public_key()):
            raise ValueError(f"recorded for key {key_id.hex()}, not this private key "
                             f"({key_id_of(priv.public_key()).hex()})")
        ck = priv.decrypt(wrapped, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                algorithm=hashes.SHA256(), label=None))
        self.k = {name: HKDF(hashes.SHA256(), 32, None,
                             f"trinet-sdenc-v1/{name}".encode()).derive(ck)
                  for name in ("video", "audio") + SIDECARS + STEREO_EXTRA}

    def video_nonce(self, frame_no, nal_idx):
        if self.alg == ALG_AES_CTR:
            return struct.pack(">IIII", 0, frame_no, nal_idx, 0)
        return struct.pack(">III", 0, frame_no, nal_idx)

    def audio_nonce(self, index):
        if self.alg == ALG_AES_CTR:
            return struct.pack(">QQ", index, 0)
        return struct.pack(">IQ", 0, index)

    def side_nonce(self, part=1):
        """Part N of a rolled sidecar (<base>.00N.wav) uses 0^4 | u64 BE (N-1)."""
        if self.alg == ALG_AES_CTR:
            return bytes(16)
        return struct.pack(">IQ", 0, part - 1)

    def xor(self, name, nonce, data, offset=0):
        """Keystream XOR starting `offset` bytes into the stream for `nonce`."""
        if self.alg == ALG_AES_CTR:   # full 128-bit big-endian counter
            blk = (int.from_bytes(nonce, "big") + offset // 16) % (1 << 128)
            dec = Cipher(algorithms.AES(self.k[name]),
                         modes.CTR(blk.to_bytes(16, "big"))).decryptor()
            dec.update(bytes(offset % 16))
        else:                         # RFC 8439: 32-bit LE block counter | 96-bit nonce
            ctr = (offset // 64).to_bytes(4, "little")
            dec = Cipher(algorithms.ChaCha20(self.k[name], ctr + nonce), mode=None).decryptor()
            dec.update(bytes(offset % 64))
        return dec.update(data) + dec.finalize()


def parse_tkey(b):
    """(key_id, wrapped, alg) from a .tkey file."""
    if len(b) < TKEY_LEN or b[:4] != b"TKEY" or b[4] != 1 or b[5] not in (1, 2):
        raise ValueError("bad .tkey")
    return b[8:16], b[16:272], b[5]


# ---------------------------------------------------------------- NALs

def unescape(b):
    out = bytearray()
    zeros = 0
    for x in b:
        if zeros >= 2 and x == 3:
            zeros = 0
            continue
        out.append(x)
        zeros = zeros + 1 if x == 0 else 0
    return bytes(out)


def nal_type(nal, hevc):
    return (nal[0] >> 1) & 0x3F if hevc else nal[0] & 0x1F


def is_clear(nal, hevc):
    t = nal_type(nal, hevc)
    return 32 <= t <= 38 if hevc else (7 <= t <= 13 or t == 15)


def key_sei(nal, hevc):
    """(key_id, wrapped, alg) if this NAL is our key SEI, else None."""
    hl = 2 if hevc else 1
    if nal_type(nal, hevc) != (39 if hevc else 6):
        return None
    p = unescape(nal[hl:hl + 400])
    if p[:3] != b"\x05\xff\x1d" or p[3:19] != KEY_SEI_UUID or p[19] != 1 or p[20] not in (1, 2):
        return None
    return p[23:31], p[31:287], p[20]


def decrypt_nal(keys, nal, hevc, vname="video"):
    hl = 2 if hevc else 1
    body = unescape(nal[hl:])
    if len(body) < 7 or body[-1] != 0x80:
        raise ValueError("not an encrypted NAL")
    frame_no, nal_idx = struct.unpack(">IH", body[:6])
    return nal[:hl] + keys.xor(vname, keys.video_nonce(frame_no, nal_idx), body[6:-1])


def filler(size, hevc):
    """A length-prefixed filler NAL occupying exactly `size` bytes."""
    hdr = bytes([38 << 1, 1]) if hevc else b"\x0c"
    n = size - 4
    if n < len(hdr) + 1:
        raise ValueError("no room for filler")
    return struct.pack(">I", n) + hdr + b"\xff" * (n - len(hdr) - 1) + b"\x80"


def split_annexb(data):
    i, n, starts = 0, len(data), []
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            break
        starts.append(j + 3)
        i = j + 3
    for k, s in enumerate(starts):
        e = starts[k + 1] - 3 if k + 1 < len(starts) else n
        while e > s and data[e - 1] == 0:
            e -= 1
        if e > s:
            yield data[s:e]


def decrypt_annexb(priv, data, hevc, keys=None, vname="video"):
    out = bytearray()
    for nal in split_annexb(data):
        ks = key_sei(nal, hevc)
        if ks:
            if keys is None:
                keys = FileKeys(priv, *ks)
            continue
        if not is_clear(nal, hevc):
            if keys is None:
                raise ValueError("encrypted NAL before any key SEI")
            nal = decrypt_nal(keys, nal, hevc, vname)
        out += b"\x00\x00\x00\x01" + nal
    return bytes(out)


# ---------------------------------------------------------------- MP4

def boxes(f, start, end):
    off = start
    while off + 8 <= end:
        f.seek(off)
        size, typ = struct.unpack(">I4s", f.read(8))
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr:
            break
        yield typ.decode("latin1"), off, off + hdr, off + size
        off += size


def child(f, start, end, name):
    for t, _, b, e in boxes(f, start, end):
        if t == name:
            return b, e
    return None


class Track:
    def __init__(self):
        self.id = 0
        self.kind = None      # "vide" / "soun"
        self.hevc = False
        self.len_size = 4
        self.samples = []     # (offset, size) in decode order
        self.trex_size = 0


def read_track(f, b, e):
    t = Track()
    tkhd = child(f, b, e, "tkhd")
    f.seek(tkhd[0])
    ver = f.read(1)[0]
    f.seek(tkhd[0] + (20 if ver == 1 else 12))
    t.id = struct.unpack(">I", f.read(4))[0]
    mdia = child(f, b, e, "mdia")
    hdlr = child(f, *mdia, "hdlr")
    f.seek(hdlr[0] + 8)
    t.kind = f.read(4).decode("latin1")
    stbl = child(f, *child(f, *mdia, "minf"), "stbl")
    stsd = child(f, *stbl, "stsd")
    entry_b = stsd[0] + 8                     # past version/flags + entry_count
    f.seek(entry_b)
    entry_size, fmt = struct.unpack(">I4s", f.read(8))
    fmt = fmt.decode("latin1")
    if t.kind == "vide":
        t.hevc = fmt in ("hvc1", "hev1")
        # VisualSampleEntry: 8 header + 78 fixed bytes, then child boxes
        cfg = child(f, entry_b + 8 + 78, entry_b + entry_size, "hvcC" if t.hevc else "avcC")
        if cfg:
            f.seek(cfg[0] + (21 if t.hevc else 4))
            t.len_size = (f.read(1)[0] & 3) + 1

    def full(name):
        r = child(f, *stbl, name)
        if not r:
            return None
        f.seek(r[0] + 4)
        return f.read(r[1] - r[0] - 4)

    stsz, stsc = full("stsz"), full("stsc")
    stco, co64 = full("stco"), full("co64")
    if stsz is None:
        return t
    fixed, cnt = struct.unpack(">II", stsz[:8])
    sizes = [fixed] * cnt if fixed else list(struct.unpack(f">{cnt}I", stsz[8:8 + 4 * cnt]))
    if co64:
        n = struct.unpack(">I", co64[:4])[0]
        chunks = list(struct.unpack(f">{n}Q", co64[4:4 + 8 * n]))
    else:
        n = struct.unpack(">I", stco[:4])[0]
        chunks = list(struct.unpack(f">{n}I", stco[4:4 + 4 * n]))
    n = struct.unpack(">I", stsc[:4])[0]
    runs = [struct.unpack(">III", stsc[4 + 12 * i:16 + 12 * i]) for i in range(n)]
    s = 0
    for ci, off in enumerate(chunks, 1):
        spc = 0
        for first, per, _ in runs:
            if first <= ci:
                spc = per
        for _ in range(spc):
            if s >= cnt:
                break
            t.samples.append((off, sizes[s]))
            off += sizes[s]
            s += 1
    return t


def read_fragments(f, fsize, tracks):
    by_id = {t.id: t for t in tracks}
    for typ, boff, b, e in boxes(f, 0, fsize):
        if typ != "moof":
            continue
        for ttyp, _, tb, te in boxes(f, b, e):
            if ttyp != "traf":
                continue
            tfhd = child(f, tb, te, "tfhd")
            f.seek(tfhd[0])
            flags = struct.unpack(">I", f.read(4))[0] & 0xFFFFFF
            tid = struct.unpack(">I", f.read(4))[0]
            base = boff
            if flags & 0x1:
                base = struct.unpack(">Q", f.read(8))[0]
            if flags & 0x2:
                f.read(4)
            if flags & 0x8:
                f.read(4)
            dflt = by_id[tid].trex_size if tid in by_id else 0
            if flags & 0x10:
                dflt = struct.unpack(">I", f.read(4))[0]
            for rtyp, _, rb, re_ in boxes(f, tb, te):
                if rtyp != "trun":
                    continue
                f.seek(rb)
                vf = struct.unpack(">I", f.read(4))[0]
                tf = vf & 0xFFFFFF
                cnt = struct.unpack(">I", f.read(4))[0]
                off = base
                if tf & 0x1:
                    off = base + struct.unpack(">i", f.read(4))[0]
                if tf & 0x4:
                    f.read(4)
                for _ in range(cnt):
                    if tf & 0x100:
                        f.read(4)
                    size = struct.unpack(">I", f.read(4))[0] if tf & 0x200 else dflt
                    if tf & 0x400:
                        f.read(4)
                    if tf & 0x800:
                        f.read(4)
                    if tid in by_id:
                        by_id[tid].samples.append((off, size))
                    off += size


def mp4_tracks(f):
    fsize = f.seek(0, 2)
    moov = child(f, 0, fsize, "moov")
    if not moov:
        raise ValueError("no moov")
    tracks = [read_track(f, b, e) for t, _, b, e in boxes(f, *moov) if t == "trak"]
    mvex = child(f, *moov, "mvex")
    if mvex:
        for t, _, b, e in boxes(f, *mvex):
            if t == "trex":
                f.seek(b + 4)
                tid, _, _, dsize = struct.unpack(">IIII", f.read(16))
                for tr in tracks:
                    if tr.id == tid:
                        tr.trex_size = dsize
        read_fragments(f, fsize, tracks)
    return tracks


def first_key_sei(f, video):
    for off, size in video.samples:
        f.seek(off)
        s = f.read(size)
        p = 0
        while p + video.len_size <= len(s):
            n = int.from_bytes(s[p:p + video.len_size], "big")
            ks = key_sei(s[p + video.len_size:p + video.len_size + n], video.hevc)
            if ks:
                return ks
            p += video.len_size + n
    return None


def decrypt_mp4(priv, src, dst, tkey=None, lane=0):
    """Decrypt src into dst. On any failure (wrong key, damaged file) dst is
    removed, so an encrypted copy is never left where decrypted output goes."""
    shutil.copyfile(src, dst)
    try:
        return _decrypt_mp4_in_place(priv, src, dst, tkey, lane)
    except BaseException:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise


def _decrypt_mp4_in_place(priv, src, dst, tkey, lane):
    vname, aname = ("video_R", "audio_R") if lane == 1 else ("video", "audio")
    with open(dst, "r+b") as f:
        tracks = mp4_tracks(f)
        video = next(t for t in tracks if t.kind == "vide")
        ks = parse_tkey(tkey) if tkey else first_key_sei(f, video)
        if not ks:
            raise ValueError(f"{src}: no .tkey and no key SEI - not an encrypted recording?")
        keys = FileKeys(priv, *ks)
        L = video.len_size
        for off, size in video.samples:
            f.seek(off)
            s = f.read(size)
            out, p = bytearray(), 0
            while p + L <= len(s):
                n = int.from_bytes(s[p:p + L], "big")
                nal = s[p + L:p + L + n]
                p += L + n
                if not nal or key_sei(nal, video.hevc):
                    continue
                if not is_clear(nal, video.hevc):
                    nal = decrypt_nal(keys, nal, video.hevc, vname)
                out += len(nal).to_bytes(L, "big") + nal
            pad = size - len(out)
            if pad:
                out += filler(pad, video.hevc) if L == 4 else b""
            if len(out) != size:
                raise ValueError(f"sample at {off}: cannot pad {pad} bytes")
            f.seek(off)
            f.write(out)
        for t in tracks:
            if t.kind != "soun":
                continue
            for i, (off, size) in enumerate(t.samples):
                f.seek(off)
                s = f.read(size)
                f.seek(off)
                f.write(keys.xor(aname, keys.audio_nonce(i), s))
    return keys


def decrypt_sidecar(keys, name, src, dst, part=1):
    data = open(src, "rb").read()
    open(dst, "wb").write(keys.xor(name, keys.side_nonce(part), data))


def stereo_parts(mp4):
    """(take base, [(eye mp4, eye suffix, lane)]) for a stereo eye file whose
    fileset key sits at <base>.tkey, else None."""
    b = mp4[:-4]
    for suf in EYE_LANE:
        if b.endswith(suf) and os.path.exists(b[:-2] + ".tkey"):
            base = b[:-2]
            eyes = [(f"{base}{e}.mp4", e, ln) for e, ln in EYE_LANE.items()
                    if os.path.exists(f"{base}{e}.mp4")]
            return base, eyes
    return None


def decrypt_stereo_take(priv, base, eyes, out_dir):
    tkey = open(base + ".tkey", "rb").read()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(base)
    done, keys = [], None
    for mp4, suf, lane in eyes:
        keys = decrypt_mp4(priv, mp4, os.path.join(out_dir, f"{stem}{suf}.mp4"), tkey, lane)
        done.append(f"{stem}{suf}.mp4")
        vts = f"{base}{suf}.vts"
        if os.path.exists(vts):
            decrypt_sidecar(keys, "vts_R" if lane == 1 else "vts", vts,
                            os.path.join(out_dir, f"{stem}{suf}.vts"))
            done.append(f"{stem}{suf}.vts")
    for ext in ("imu", "tel", "wav"):
        p = f"{base}.{ext}"
        if os.path.exists(p):
            decrypt_sidecar(keys, ext, p, os.path.join(out_dir, f"{stem}.{ext}"))
            done.append(f"{stem}.{ext}")
    part = 2
    while os.path.exists(f"{base}.{part:03d}.wav"):
        name = f"{stem}.{part:03d}.wav"
        decrypt_sidecar(keys, "wav", f"{base}.{part:03d}.wav", os.path.join(out_dir, name), part)
        done.append(name)
        part += 1
    return done


def decrypt_take(priv, mp4, out_dir):
    st = stereo_parts(mp4)
    if st:
        return decrypt_stereo_take(priv, st[0], st[1], out_dir)
    base = mp4[:-4]
    tk = base + ".tkey"
    tkey = open(tk, "rb").read() if os.path.exists(tk) else None
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(base)
    keys = decrypt_mp4(priv, mp4, os.path.join(out_dir, stem + ".mp4"), tkey)
    done = [stem + ".mp4"]
    for ext in SIDECARS:
        p = f"{base}.{ext}"
        if os.path.exists(p):
            decrypt_sidecar(keys, ext, p, os.path.join(out_dir, f"{stem}.{ext}"))
            done.append(f"{stem}.{ext}")
    # The .wav rolls to <base>.002.wav, ... at its 4 GiB cap; each part has its own nonce.
    part = 2
    while os.path.exists(f"{base}.{part:03d}.wav"):
        name = f"{stem}.{part:03d}.wav"
        decrypt_sidecar(keys, "wav", f"{base}.{part:03d}.wav", os.path.join(out_dir, name), part)
        done.append(name)
        part += 1
    return done


def find_takes(path):
    if os.path.isfile(path):
        return [(path, "")]
    takes = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if not fn.endswith(".mp4"):
                continue
            full = os.path.join(root, fn)
            if os.path.exists(full[:-4] + ".tkey"):
                takes.append((full, os.path.relpath(root, path)))
            elif fn[:-4].endswith("_L") or (fn[:-4].endswith("_R") and
                                            fn[:-6] + "_L.mp4" not in files):
                # stereo: one entry per take (the _L file, or _R alone)
                if os.path.exists(full[:-6] + ".tkey"):
                    takes.append((full, os.path.relpath(root, path)))
    return takes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("keygen", help="create an RSA-2048 key pair")
    g.add_argument("priv_pem")
    g.add_argument("pub_tvk")
    g.add_argument("--passphrase", action="store_true", help="encrypt priv.pem (prompted)")
    x = sub.add_parser("export-pubkey", help="write the camera-format public key")
    x.add_argument("priv_pem")
    x.add_argument("pub_tvk")
    k = sub.add_parser("export-signing-key",
                       help="write an unencrypted PKCS#8 copy of priv.pem for the phone app "
                            "(to change or remove the camera's key; keep it as secret as priv.pem)")
    k.add_argument("priv_pem")
    k.add_argument("out_pem")
    d = sub.add_parser("decrypt", help="decrypt a take, folder or whole card")
    d.add_argument("path")
    d.add_argument("--key", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--annexb", action="store_true",
                   help="input is a raw encrypted .h264/.h265 stream (test/debug)")
    d.add_argument("--hevc", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "keygen":
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        enc = serialization.NoEncryption()
        if a.passphrase:
            pw = getpass.getpass("new passphrase: ").encode()
            if pw != getpass.getpass("again: ").encode():
                sys.exit("passphrases differ")
            enc = serialization.BestAvailableEncryption(pw)
        fd = os.open(a.priv_pem, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(priv.private_bytes(serialization.Encoding.PEM,
                                        serialization.PrivateFormat.PKCS8, enc))
        open(a.pub_tvk, "wb").write(pubkey_to_tvk(priv.public_key()))
        print(f"key id {key_id_of(priv.public_key()).hex()}: {a.priv_pem} (SECRET), {a.pub_tvk} (camera)")
        return 0
    priv = load_private(a.key if a.cmd == "decrypt" else a.priv_pem)
    if a.cmd == "export-signing-key":
        # Some phones cannot open a passphrase-protected (PBES2) PKCS#8 file. The
        # app only SIGNS with it, on the phone, but it is the same secret as
        # priv.pem: copy it to the phone and delete it afterwards.
        fd = os.open(a.out_pem, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(priv.private_bytes(serialization.Encoding.PEM,
                                        serialization.PrivateFormat.PKCS8,
                                        serialization.NoEncryption()))
        print(f"key id {key_id_of(priv.public_key()).hex()} -> {a.out_pem} "
              f"(SECRET, unencrypted: delete it after use)")
        return 0
    if a.cmd == "export-pubkey":
        open(a.pub_tvk, "wb").write(pubkey_to_tvk(priv.public_key()))
        print(f"key id {key_id_of(priv.public_key()).hex()} -> {a.pub_tvk}")
        return 0
    if a.annexb:
        os.makedirs(a.out, exist_ok=True)
        dst = os.path.join(a.out, os.path.basename(a.path))
        open(dst, "wb").write(decrypt_annexb(priv, open(a.path, "rb").read(), a.hevc))
        print(dst)
        return 0
    takes = find_takes(a.path)
    if not takes:
        sys.exit(f"no encrypted takes (*.mp4 with a .tkey) under {a.path}")
    bad = 0
    for mp4, rel in takes:
        out_dir = os.path.join(a.out, rel)
        before = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
        try:
            done = decrypt_take(priv, mp4, out_dir)
            print(f"ok   {mp4}: {', '.join(done)}")
        except Exception as ex:  # keep going: one damaged take must not stop a card
            bad += 1
            # A failed take leaves nothing behind (not even the other eye or a
            # sidecar): partial output could pass for a decrypted take.
            for fn in (set(os.listdir(out_dir)) - before) if os.path.isdir(out_dir) else ():
                try:
                    os.remove(os.path.join(out_dir, fn))
                except OSError:
                    pass
            print(f"FAIL {mp4}: {ex}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

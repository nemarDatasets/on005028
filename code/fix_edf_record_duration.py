#!/usr/bin/env python3
"""Repair the EDF data-record layout of on005028 (BCI2000 -> EDF export).

Every ``*_eeg.edf`` in this dataset declares ONE data record of 1 s that holds
all N samples of each signal, so readers take the sampling rate to be N Hz
(sub-01_ses-test_run-1: 84,640 Hz for a 330.6 s recording). The BCI2000 source
file next to each EDF (``*_eeg.dat``) states ``SamplingRate= 256Hz`` for all
105 recordings.

This script rewrites each EDF into records of g samples, g = gcd(N, 256), and
duration g/256 s, so the declared rate is exactly 256 Hz. Only the record
layout and three header fields change (number of data records, record
duration, samples per record); every sample value is kept, in the same order.
Files whose layout differs from the one described (several records, unequal
samples per signal, an EDF+ annotations signal, or a .dat rate other than
256 Hz) are left untouched and reported.

usage: python code/fix_edf_record_duration.py [--check] <edf> [<edf> ...]
  --check  only report; write nothing.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

FS = 256


def _field(buf: bytes, a: int, n: int) -> str:
    return buf[a : a + n].decode("ascii").strip(" \x00")


def _put(buf: bytearray, a: int, n: int, value: str) -> None:
    text = value.ljust(n).encode("ascii")
    if len(text) != n:
        raise ValueError(f"{value!r} does not fit {n} characters")
    buf[a : a + n] = text


def dat_rate(dat: Path) -> float | None:
    head = dat.read_bytes()[:65536].decode("latin1")
    m = re.search(r"SamplingRate= *([0-9.]+)", head)
    return float(m.group(1)) if m else None


def repair(edf: Path, *, check: bool) -> str:
    raw = edf.read_bytes()
    ns = int(_field(raw, 252, 4))
    hdr = 256 * (ns + 1)
    n_rec = int(_field(raw, 236, 8))
    dur = float(_field(raw, 244, 8))
    labels = [_field(raw, 256 + 16 * i, 16) for i in range(ns)]
    spr = [int(_field(raw, 256 + 216 * ns + 8 * i, 8)) for i in range(ns)]
    dat = edf.with_suffix(".dat")
    rate = dat_rate(dat) if dat.exists() else None
    if n_rec != 1 or dur != 1 or len(set(spr)) != 1 or "EDF Annotations" in labels:
        return "skip: layout differs"
    if rate != FS:
        return f"skip: .dat SamplingRate {rate}"
    n = spr[0]
    if len(raw) != hdr + 2 * ns * n:
        return "skip: size does not match header"
    g = math.gcd(n, FS)
    duration = f"{g / FS:g}"
    if check:
        return f"would fix: {n} samples -> {n // g} records x {g} samples, {duration} s"
    data = np.frombuffer(raw, dtype="<i2", offset=hdr).reshape(ns, n // g, g)
    out = bytearray(raw[:hdr])
    _put(out, 236, 8, str(n // g))
    _put(out, 244, 8, duration)
    for i in range(ns):
        _put(out, 256 + 216 * ns + 8 * i, 8, str(g))
    # Signal-major (one record) -> record-major (record, signal, sample).
    out += data.transpose(1, 0, 2).tobytes()
    tmp = edf.with_suffix(".edf.tmp")
    tmp.write_bytes(bytes(out))
    tmp.replace(edf)
    return f"fixed: {n} samples, {n // g} records x {g}, {n / FS:.3f} s"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    files = [Path(a) for a in argv if a != "--check"]
    for edf in files:
        print(edf, repair(edf, check=check))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

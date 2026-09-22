"""Assert every numeric claim in the run 7 lock ruling against results/*.txt.

A hand check found one wrong figure (N=28 correct_endpoints, 0.0% vs the file's
0.5%), so the whole ruling is now verified mechanically rather than by reading.
Run from anywhere: paths are repo-root relative.

    python diagnostics/check_run7_ruling_claims.py
"""
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

M9B = open(os.path.join(_ROOT, "results", "m9b_run5.txt"), encoding="utf-8").read()
OOD = open(os.path.join(_ROOT, "results", "ood_diagnostics_run5.txt"),
           encoding="utf-8").read()

fails = []


def block(header_pat, text):
    """Metric dict for the first block whose header matches."""
    m = re.search(header_pat, text)
    if not m:
        return None
    tail = text[m.end():]
    nxt = re.search(r"\n\s*\n", tail)
    seg = tail[: nxt.start()] if nxt else tail
    out = {}
    for name, val in re.findall(r"(\w+)\s*:\s*\d+\s*\(\s*([\d.]+)%\)", seg):
        out[name] = float(val)
    d = re.search(r"Density:\s*([\d.]+)", tail[:400])
    if d:
        out["density"] = float(d.group(1))
    h = re.search(r"Mean Dijkstra hops\s*:\s*([\d.]+)", tail[:600])
    if h:
        out["hops"] = float(h.group(1))
    return out


def rng(vals):
    return min(vals), max(vals)


def check(label, got, want):
    ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {label}: got {got}, ruling says {want}")
    if not ok:
        fails.append(label)


# --- size sweep, length fixed at 184, N=25..50 ---
sweep = {}
for n in (25, 30, 35, 40, 45, 50):
    b = block(rf"Size Gen: N={n}, E=60 \(len=184\)", M9B)
    assert b, f"missing sweep block N={n}"
    sweep[n] = b

lo, hi = rng([sweep[n]["correct_endpoints"] for n in sweep])
check("sweep correct_endpoints range", (lo, hi), (79.5, 95.8))
lo, hi = rng([sweep[n]["valid_and_optimal"] for n in sweep])
check("sweep valid_and_optimal range", (lo, hi), (26.2, 63.0))
check(
    "sweep edges_valid sequence N=25..50",
    [sweep[n]["edges_valid"] for n in (25, 30, 35, 40, 45, 50)],
    [80.8, 63.5, 51.5, 41.2, 34.0, 35.2],
)

# --- test 2: N=20, E=66/69/72, lengths 202/211/220 ---
t2 = {}
for e, ln in ((66, 202), (69, 211), (72, 220)):
    b = block(rf"N=20, E={e} \(len={ln}\)", M9B)
    assert b, f"missing test2 block E={e}"
    t2[e] = b
check("test2 correct_endpoints", [t2[e]["correct_endpoints"] for e in (66, 69, 72)],
      [3.0, 0.2, 0.2])
check("test2 valid_and_optimal", [t2[e]["valid_and_optimal"] for e in (66, 69, 72)],
      [1.2, 0.0, 0.2])

# --- test 3: N=20, E=45..60, lengths 139..184 ---
t3 = {}
for e, ln in ((45, 139), (48, 148), (51, 157), (54, 166), (57, 175), (60, 184)):
    b = block(rf"N=20, E={e} \(len={ln}\)", M9B)
    assert b, f"missing test3 block E={e}"
    t3[e] = b
lo, hi = rng([t3[e]["correct_endpoints"] for e in t3])
check("test3 correct_endpoints range", (lo, hi), (94.2, 98.5))

# --- test 1: N=7, E=12/13/18/19 ---
t1 = {}
for e, ln in ((12, 40), (13, 43), (18, 58), (19, 61)):
    b = block(rf"N=7, E={e} \(len={ln}\)", M9B)
    assert b, f"missing test1 block E={e}"
    t1[e] = b
check("N=7 E=19 correct_endpoints", t1[19]["correct_endpoints"], 0.0)
check("N=7 E=19 valid_and_optimal", t1[19]["valid_and_optimal"], 0.0)
check("N=7 E=18 correct_endpoints", t1[18]["correct_endpoints"], 99.2)
check("N=7 E=18 valid_and_optimal", t1[18]["valid_and_optimal"], 93.0)
check("N=7 E=18 density", t1[18]["density"], 0.857)
check("N=7 E=19 density", t1[19]["density"], 0.905)
check("N=7 E=18 hops", t1[18]["hops"], 1.52)
check("N=7 E=19 hops", t1[19]["hops"], 1.46)

# --- OOD size table: correct_endpoints baseline for the success criterion ---
ood = {}
for line in OOD.splitlines():
    m = re.match(r"N=(\d+) \(\d+ graphs\).*?correct_ends:\s*\d+\s*\(\s*([\d.]+)%\)", line)
    if m:
        ood[int(m.group(1))] = float(m.group(2))
want = {21: 53.0, 22: 4.0, 23: 1.5, 24: 0.0, 25: 0.5, 26: 0.0, 28: 0.5, 30: 0.0}
for n, w in want.items():
    check(f"OOD correct_ends N={n}", ood.get(n), w)
check("OOD correct_ends N=20", ood.get(20), 97.5)

# --- N=25 at its own locked length: V&O 0.5%, and the 126x gap ---
v25 = None
for line in OOD.splitlines():
    m = re.match(r"N=25 \(\d+ graphs\).*?V&O:\s*\d+\s*\(\s*([\d.]+)%\)", line)
    if m:
        v25 = float(m.group(1))
check("OOD N=25 valid_and_optimal at locked length", v25, 0.5)
check("126x gap (sweep N=25 V&O / locked-length N=25 V&O)",
      round(sweep[25]["valid_and_optimal"] / v25), 126)

# --- trained region, and membership of every row the ruling cites ---
trained_E = {min(3 * n, n * (n - 1) // 2) for n in range(5, 21)}
trained_len = {3 * e + 4 for e in trained_E}
marker_pos = {3 * e for e in trained_E}

check("trained E count is sixteen", len(trained_E), 16)
check("trained E set", sorted(trained_E),
      [10, 15, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60])
check("trained length set", sorted(trained_len),
      [34, 49, 67, 76, 85, 94, 103, 112, 121, 130, 139, 148, 157, 166, 175, 184])
check("max trained length", max(trained_len), 184)
check("N=7 trained pair is (E=21, len=67)",
      (min(3 * 7, 21), 3 * min(3 * 7, 21) + 4), (21, 67))

# Test 1 rows: ruling claims none of these lengths or E values is trained.
t1_rows = {12: 40, 13: 43, 18: 58, 19: 61}
for e, ln in t1_rows.items():
    check(f"test1 E={e} length is 3E+4", 3 * e + 4, ln)
    check(f"test1 E={e} length NOT trained", ln in trained_len, False)
    check(f"test1 E={e} E NOT trained", e in trained_E, False)
check("test1 correct_endpoints E=12/13/18/19",
      [t1[e]["correct_endpoints"] for e in (12, 13, 18, 19)],
      [76.5, 87.0, 99.2, 0.0])
check("test1 valid_and_optimal E=12/13/18/19",
      [t1[e]["valid_and_optimal"] for e in (12, 13, 18, 19)],
      [60.2, 78.2, 93.0, 0.0])
# The divisibility-by-3 reading: E=18 multiple / E=19 not, but E=12 multiple
# scores LOWER than E=13 which is not. Ruling says this contradicts the reading.
check("E=12 % 3", 12 % 3, 0)
check("E=13 % 3", 13 % 3, 1)
check("E=18 % 3", 18 % 3, 0)
check("E=19 % 3", 19 % 3, 1)
check("divisibility reading contradicted (E=12 multiple scores below E=13)",
      t1[12]["correct_endpoints"] < t1[13]["correct_endpoints"], True)

# Marker positions: ruling says 54/56 (E=18) and 57/59 (E=19) are all untrained.
check("trained marker positions", sorted(marker_pos),
      [30, 45, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180])
for p in (54, 56, 57, 59):
    check(f"marker position {p} NOT trained", p in marker_pos, False)

# Test 3 rows are inside the trained region on BOTH axes; test 2 outside on both.
for e in (45, 48, 51, 54, 57, 60):
    check(f"test3 E={e} trained on both axes",
          (e in trained_E, 3 * e + 4 in trained_len), (True, True))
for e in (66, 69, 72):
    check(f"test2 E={e} untrained on both axes",
          (e in trained_E, 3 * e + 4 in trained_len), (False, False))

# Locked-rule pairs for the OOD sizes the ruling names.
for n, e_want, len_want in ((21, 63, 193), (25, 75, 229), (50, 150, 454)):
    e = min(3 * n, n * (n - 1) // 2)
    check(f"locked N={n} -> (E, len)", (e, 3 * e + 4), (e_want, len_want))
    check(f"locked N={n} outside trained region",
          (e in trained_E, 3 * e + 4 in trained_len), (False, False))
check("N=50 edge block spans positions 4..453", (4, 3 * 150 + 3), (4, 453))
check("PositionalEncoding max_len 5000 covers 454", 454 <= 5000, True)

print()
print(f"{len(fails)} FAILURES: {fails}" if fails else "all claims in the ruling verified")
sys.exit(1 if fails else 0)

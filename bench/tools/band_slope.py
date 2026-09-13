#!/usr/bin/env python3
"""Fit a swept benchmark band as intercept + slope per arm.

usage: band_slope.py RESULTS [BAND] [--param chunks] [--stat col2|median_us|mean_us] [--ref cpp-cow]

RESULTS holds frsr_roaring_bench result lines ("name<TAB>X us/op<TAB>checksum=...<TAB>field=value..."), for example
from several runs or from bench/tools/run_isolated.sh. BAND (default EnvelopePerChunk) selects entries whose name
contains it; --param names the swept parameter in the entry name (default chunks). A result line reports
microseconds per op, where one op is one unit of the swept parameter, so the time of one result is us/op * param.
Per arm the script takes the median over repeated lines of each point, fits
time_per_result = intercept + slope * param by least squares, and prints both in nanoseconds with each arm's ratio
to the reference arm. The slope is the cost per unit of the swept parameter; the intercept is the fixed cost per
result.
"""
import collections
import re
import statistics
import sys

args = list(sys.argv[1:])


def option(name, default):
    if name in args:
        i = args.index(name)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


param = option("--param", "chunks")
stat = option("--stat", "col2")
ref = option("--ref", "cpp-cow")
if not args:
    sys.exit(__doc__)
path = args[0]
band = args[1] if len(args) > 1 else "EnvelopePerChunk"
label = re.compile(r"^([a-z0-9-]+)(?=[A-Z])")
param_re = re.compile(r"(?:^|/)" + re.escape(param) + r"=(\d+)")

points = collections.defaultdict(lambda: collections.defaultdict(list))  # arm -> param value -> [us per result]
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        cols = line.rstrip("\r\n").split("\t")
        if len(cols) < 2 or "us/op" not in cols[1] or band not in cols[0] or cols[0].endswith("#aa"):
            continue
        parts = cols[0].split("/")
        arm = label.match(parts[1]) if len(parts) > 1 else None
        value = param_re.search(cols[0])
        if not arm or not value:
            continue
        fields = dict(c.split("=", 1) for c in cols[2:] if "=" in c)
        us = float(cols[1].split()[0]) if stat == "col2" else float(fields[stat])
        n = int(value.group(1))
        points[arm.group(1)][n].append(us * n)


def fit(per_point):
    xs = sorted(per_point)
    ys = [statistics.median(per_point[x]) for x in xs]
    if len(xs) < 2:
        return float("nan"), float("nan"), xs
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope, xs


fits = {arm: fit(per_point) for arm, per_point in points.items()}
if not fits:
    sys.exit("no matching result lines for band " + band + " with " + param + "= in " + path)
ref_intercept, ref_slope, _ = fits.get(ref, (float("nan"), float("nan"), []))
print(f"band {band}: time per result = intercept + slope * {param} (ns), ratios to {ref}")
print(f"{'arm':<16}{'points':>7}{'intercept':>11}{'slope':>9}{'int/ref':>9}{'slope/ref':>10}")
for arm in sorted(fits):
    intercept, slope, xs = fits[arm]
    int_ratio = intercept / ref_intercept if ref_intercept else float("nan")
    slope_ratio = slope / ref_slope if ref_slope else float("nan")
    print(f"{arm:<16}{len(xs):>7}{intercept * 1000:>11.2f}{slope * 1000:>9.3f}{int_ratio:>9.3f}{slope_ratio:>10.3f}")

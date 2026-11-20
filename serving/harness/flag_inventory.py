"""Every server flag in the pinned tree, split by the judge's denied list (rules_live/serving_<date>.json).
  python3 -I harness/flag_inventory.py rules_live/serving_2026-11-13.json > research/flag_inventory_2026-11-13.txt"""
import json, re, glob, fnmatch, os, sys
d = json.load(open(sys.argv[1]))
denied = [f for v in d["launch_flags"]["denied"].values() for f in v]
judge = [a for a in d["launch"]["args"] if a.startswith("--")]
def is_denied(flag): return any(fnmatch.fnmatch(flag, p) for p in denied)
tot = den = 0
for f in sorted(glob.glob("sglang_stock/python/sglang/srt/arg_groups/fields/*.py")):
    src = open(f).read()
    names = ["--" + m.group(1).replace("_", "-") for m in re.finditer(r"^    ([a-z][a-z0-9_]*): A\[", src, re.M)]
    allowed = [n for n in names if not is_denied(n)]
    tot += len(names); den += len(names) - len(allowed)
    print(f"## {os.path.basename(f)}: {len(allowed)} allowed of {len(names)}")
    for n in allowed: print("  ", n, "(judge sets it; override = changes the judge's value)" if n in judge else "")
print(f"\nTOTAL {tot} flags, {den} denied by pattern, {tot-den} allowed")
print("denied patterns:", denied)
print("judge launch args:", judge)

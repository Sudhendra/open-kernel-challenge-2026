"""Chain experiments onto a running Modal session container (queue mode of modal_serve.py::session --queue <dir>).

  python3 harness/modal_queue.py push <queue> <name> --variants "a:SRC=patched;b:SRC=w" [--rungs 1x1x2,4x1x2] \
      [--prof-conc "1 4"] [--max-min 60] [--extra-args "--hicache-size 42"] [--server-env-extra K=V,K=V] [--shape judge|short] [--corpus DIR]
      [--prof-env WARM_EXTEND=12,WARM_SKIP=0] [--proxy-client ids]   # PROF=warm knobs (profile_session.sh header); WARM_* names in WARM_KEYS only
  python3 harness/modal_queue.py ls <queue>          # pending / done items
  python3 harness/modal_queue.py stop <queue>        # the container exits after the current item
The queue dir lives on the okc-muse-weights volume (e.g. queue/prof). Items run in name order (prefix 01_, 02_ ...).
Only trees mounted at launch (OKC_PATCHED / OKC_EXTRA_TREES) are available: SRC=<name> must be one of them.
"""
import argparse, os, subprocess, sys, tempfile
VOL = "okc-muse-weights"
# profile_session.sh PROF=warm knobs an item may set (modal_serve.py _serve_queue passes exactly these through)
WARM_KEYS = ("WARM_TURNS", "WARM_EXTEND", "WARM_SKIP", "WARM_MAX_STEPS", "WARM_MIN_CACHED", "WARM_WITH_STACK", "WARM_PYSPY", "WARM_PYSPY_S", "WARM_PYSPY_ARGS", "WARM_WAIT_S")
def prof_env_lines(spec):
    """'WARM_EXTEND=12,WARM_SKIP=4' -> ['WARM_EXTEND=12', 'WARM_SKIP=4']; exits on a key outside WARM_KEYS or a value with a comma."""
    out = []
    for kv in filter(None, (x.strip() for x in spec.split(","))):
        k, sep, v = kv.partition("=")
        if not sep or k not in WARM_KEYS: sys.exit(f"--prof-env: {kv!r}: want K=V with K in {', '.join(WARM_KEYS)}")
        out.append(f"{k}={v}")
    return out
def vol(*args, check=True): return subprocess.run(["modal", "volume", *args, VOL] if args[0] == "ls" else ["modal", "volume", args[0], VOL, *args[1:]], check=check, capture_output=True, text=True)
def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("queue"); p.add_argument("name"); p.add_argument("--variants", required=True)
    p.add_argument("--rungs", default=""); p.add_argument("--prof-conc", default=""); p.add_argument("--max-min", type=int, default=0)
    p.add_argument("--extra-args", default=""); p.add_argument("--server-env-extra", default=""); p.add_argument("--stock-reuse", default="", help="run id under out/ to reuse (default: the container's first run)")
    p.add_argument("--shape", default="", choices=["", "judge", "short"], help="PROXY_SHAPE (default: the container's, judge); must match the reused stock")
    p.add_argument("--corpus", default="", help="PROXY_CORPUS: ladder.py --corpus directory inside the image (default: the container's)")
    p.add_argument("--proxy-client", default="", choices=["", "chat", "ids"], help="PROXY_CLIENT: ids = /generate with client-built input_ids (the judge's client); default chat")
    p.add_argument("--prof-env", default="", help="PROF=warm knobs, K=V,K=V with K in WARM_KEYS (default: profile_session.sh defaults)")
    l = sub.add_parser("ls"); l.add_argument("queue")
    st = sub.add_parser("stop"); st.add_argument("queue")
    a = ap.parse_args()
    if a.cmd == "push":
        if not a.name.replace("_", "").replace("-", "").isalnum(): sys.exit("name must be alphanumeric/_/-")
        lines = [f"VARIANTS={a.variants}"]
        for k, v in [("PROXY_RUNGS", a.rungs), ("PROF_CONC", a.prof_conc), ("EXTRA_ARGS", a.extra_args), ("SERVER_ENV_EXTRA", a.server_env_extra), ("STOCK_REUSE", a.stock_reuse), ("PROXY_SHAPE", a.shape), ("PROXY_CORPUS", a.corpus), ("PROXY_CLIENT", a.proxy_client)]:
            if v: lines.append(f"{k}={v}")
        if a.max_min: lines.append(f"MAX_MIN={a.max_min}")
        lines += prof_env_lines(a.prof_env)
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f: f.write("\n".join(lines) + "\n"); tmp = f.name
        r = subprocess.run(["modal", "volume", "put", VOL, tmp, f"{a.queue}/{a.name}.env", "--force"], capture_output=True, text=True)
        os.unlink(tmp); print(r.stdout.strip() or r.stderr.strip()); print("\n".join(lines))
    elif a.cmd == "ls":
        for d in (a.queue, f"{a.queue}/done"):
            r = subprocess.run(["modal", "volume", "ls", VOL, d], capture_output=True, text=True); print(f"== {d}\n{r.stdout.strip() or r.stderr.strip()}")
    elif a.cmd == "stop":
        with tempfile.NamedTemporaryFile("w", delete=False) as f: f.write("stop\n"); tmp = f.name
        r = subprocess.run(["modal", "volume", "put", VOL, tmp, f"{a.queue}/STOP", "--force"], capture_output=True, text=True); os.unlink(tmp); print(r.stdout.strip() or r.stderr.strip())
if __name__ == "__main__": main()

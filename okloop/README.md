# okloop — one objective, one gate, many cheap attempts

Objective: muse_decode_layer local judge score (pod/bench.py: 12 shapes, cold-L2 graph replay, paired, geomean of 6 loads).
Target: rank 40 on the board (2.296 on Nov 1; plan for 2.32).

Shared state (all of it): best/submission.py (current best = board file), LEARNINGS.md (keeps + dead list), queue/ (ports).
An attempt = a fresh provider process in attempts/<id>/ that reads best.py, lane.md, LEARNINGS.md and must return candidate.py + NOTE.md.
The gate = gate.sh: lint -> cold compile < 270 s -> all 13 tests -> two independent paired timings on box A; keep iff both beat
max(0.3%, 2 x A/A band). A keep replaces best/, is committed, appended to LEARNINGS.md, synced to the box, and submitted per the policy in config.env.

    ./okloop box setup     # once per box (judge container, image, caches)      ./okloop box sync   # push track files + best
    ./okloop aa            # measure the A/A band (10 min, do this once per box start)
    ./okloop start | stop | pause | resume | status | log
    ./okloop gate FILE [stage]   ./okloop attempt LANE PROVIDER   ./okloop approve

Lanes (config.env LANES): attn, moe, glue (Claude/Codex, kernel edits), ports (queue/), research (every 3 h, web -> queue/).
Box: Nebius spot H200 SXM VM, docker container `judge` from the official SGLang image, repo at ~/okc, GPU jobs serialized by flock.
Logs: state/log.tsv (one line per event), attempts/<id>/{agent.log,gate.log,compile.log,correct.log,time1.log,time2.log,result.json}, logs/loop.out.
Submissions: state/submits.tsv. SUBMIT=auto sends the first AUTO_SLOTS keeps/day that are SUBMIT_MIN_GAIN over the last submitted file; the rest wait for `okloop approve`.

Second research scanner (DeepSeek via opencode, independent process, same queue/): `./okloop research2 start|stop|status`; log lane `research-ds` in state/log.tsv, output logs/research-ds.out.

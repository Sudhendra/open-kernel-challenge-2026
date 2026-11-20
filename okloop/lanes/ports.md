# Nov1 frontier baseline and research priority

The board and local seed are now the public-integrated frontier decoder: official2.2512/rank105, independent fresh-input local gains+2.53%/+2.13%. `best/frontier-promotion.json` records the hashes. The loop remains paused. The old box-local score and A/A files describe the previous source and must be recalibrated through ogpu before relying on them.

Read `research/2026-11-01_breakthrough/README.md` in the repository root. MFIN, SC6, register-A projection and public per-shape improvements are now already in best. Reapplying an exhausted historical port is not progress. Prioritize `queue/frontier_q8_fourwg.md`; `queue/frontier_moe_d64_budget.md` is a budget-first second choice. Both are explicitly unbuilt hypotheses and may end in a measured veto rather than a candidate.

All GPU work must go through ogpu. Use the repository `research/2026-11-01_breakthrough/ogpu_json.py` wrapper for JSON commands so provider error tracebacks cannot expose credentials. The legacy `./gate` / box.sh wrappers use direct SSH and are not the approved execution path. Stage small experiments with the required reference/check helpers, use CPU compile/resource gates, then cheap correctness, then H200 timing. Final measurements must vary inputs/routing, reverse capture order, and use true AB/BA replay order. The shared reconstruction harness order was fixed on Nov1, but its default fixed replay input is still inadequate for claims about fresh routing. The frontier fresh_bench.py and portfolio_bench.py show the validated protocols.

Do not resume the loop or start another agent from this instruction. Respect the existing PAUSE marker. Once the user resumes work, test a new premise rather than running the same exhausted queue again.

---

# Lane: ports — carry measured wins from the standalone tracks into the layer, one per attempt
Why: on lxe's loop 24 of the first 35 keeps were ports or combinations. Our standalone attention and MoE kernels share the vjeux lineage with the in-layer ones, so their diffs transplant.

How: ./queue/ has one .md per item with the source diff and what it changed. Pick the item with the highest "measured delta" that is not marked DONE or DEAD in LEARNINGS.md.
Read the item's diff. Find the same code in best.py (the in-layer kernel is a renamed copy: `muse_att_v298w1` for attention, `_moe_build` defines for MoE). Apply only that diff. Do not bring anything else from the standalone file.
If the in-layer kernel already contains the change, note it as "already present: <item>" in NOTE.md and MOVE ON to the next best item (also skip items listed as DONE in LEARNINGS.md). Do not stop until you have applied one item that is not already present; only if every remaining item is already present or DEAD, write that in NOTE.md and stop.

Rules of thumb:
- Attention items affect FULL shapes; MoE flag items affect the M=16/32 builds; decode items (pdl128_re, stack2, wgsrb, mid32) are whole-layer config diffs.
- Combine two queue items in one attempt only if both are 2-line flag changes.
- After you apply the diff, run `./gate --stage correct candidate.py`. Correctness differences between the standalone kernel and the in-layer one (k_scale, gating, q_len 8 path) are the usual failure.

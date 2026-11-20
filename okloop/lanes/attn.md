# Lane: attn — the in-layer attention path
Owns: everything from qkv output to the o-projection input: QK-norm, RoPE, FP8 KV write, split-KV attention, combine, rmsnorm*sigmoid(g).
Why: full-attention shapes are 30-55% of the full loads (d16f attn 89 us of 210, d32f 170 of 304, v16f 102 of 346, v32f 187 of 492).
Sliding shapes spend only 5-15 us here; do not spend the attempt on sliding unless the change is free.

Anchors in best.py (line numbers approximate):
- `_ATT7 = load_inline(name="muse_att_v298w1_...")` ~3718: the fused CUDA attention (ATT7_CUDA_SRC above it); knobs OKC_FLATCOOP/OKC_PDL/OKC_FLAT1B/OKC_PLAIN8/OKC_F2/OKC_S2/OKC_W0_* at ~3761-3769.
- Triton `_attn_split` ~10211 and `_attn_combine` ~10289: the split-KV fallback path and its combine.
- `_attn_cfg(B, L, sliding)` ~11013: per-shape configuration (nsplit, tiles, which kernel). `ATT_HAS_PDL` ~11095. `_attn_selftest` ~11137.
- kernel_fn (last definition wins, ~13083) -> look for the attention call site and what it launches per shape.

Ideas not on the dead list (pick ONE, smallest first):
1. Per-shape nsplit / tile configuration for the FULL shapes at B=16/32 and q_len 8 (v16f/v32f): the standalone track found 264-CTA equal splits optimal there; check what _attn_cfg picks in-layer for each of the 6 full shapes and align it.
2. Combine step: is `_attn_combine` (or the CUDA merge) a separate launch at any full shape? Fold it into the attention kernel's tail or into the o-projection prologue where the layer already has a cgemv/sk2 launch.
3. KV write + RoPE for q_len 8 (verify shapes): check the new-token path is one launch and not re-reading the page table.
4. Page-table / index prefetch: pfband-style L2 prefetch of the page-table band during the seq_lens trip (standalone: +0.2%, free on sliding). See queue/ for the standalone diff.
5. Verify the in-layer attention passes k_scale/v_scale == 1 fast path on scored shapes (tests use 1/64 only).

Measure first: `./gate --stage correct candidate.py` after any change to the CUDA source. The combine/merge path has the most correctness traps (rel-L2 tolerances in LEARNINGS).

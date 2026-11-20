# Lane: moe — the in-layer NVFP4 MoE at M = 16 and 32 (the worst loads)
Why: decode32 (1.72-1.85x) and decode16 (~2.0x) are the weakest loads and the geomean rewards fixing the worst load most.
At d32s the MoE is 86 us of 150; at d16s 71 of 135. At v16/v32 it is 148-228 us (MAIN/WGS builds, FT_T=64 units).

Anchors in best.py:
- MoE builds: `_moe_build(defines)` ~6739, dispatch `_moe(M)` ~6748 and the later override `_moe(M=None, state=None)` ~13144; MID `_moe_mid` ~6773, TINY2 ~6785, TINY ~6792.
- Define sets: MOE_DEFINES ~6674 (MAIN/WGS), MOE_MAIN_YBFM_FLAGS ~6676 (the kept ybf16/finsmem/nofence), MOE_TINY_DEFINES ~6677 (M<=4), MOE_TINY2 ~6679 (M<=8), MOE_MID_DEFINES ~6681, MAIN_FALLBACK_DEFINES ~6948, `_MA_MAIN_NS` ~6705, MOE_HAS_PDL ~6697.
- Routing prologue and combine live inside the MoE kernel (one launch). KPERM3 is the alygg77 permutation.

Dead list covers: every WGS-body sync/epilogue/helper-warp change, ring depth, L2 prefetch, ping-pong, FP8 MMA, mid32 alone. Do not go there.

Ideas not on the dead list (pick ONE):
1. The d16/d32 build choice: which build (TINY2/MID/MAIN/WGS32) runs at M=16 and M=32 in-layer today, with which -D flags? Compare the MoE board's current winning flag set for M=16/32 (muse_moe_w4a16 submission.py defines) against ours and port a differing flag. Flags are a 2-line diff; the queue has c9_mid_ns4 and c_ybfm_wgs as examples.
2. FT_T per M: at M=16/32 the number of touched experts is 12-15 and no expert exceeds 32 tokens (routing probe), so unit splitting is pure overhead there; check the unit size the MID/WGS32 build uses and whether a 32-token unit with fewer unit-ends applies at M<=32 only.
3. M=16 with WGS32 build (`-DOKC_FT_T=32`, 17<=M<=32 today): try the same policy for 9<=M<=16 if MID re-streams weights there (the "9-16t re-stream" note in the MoE track's C5).
4. MoE launch prologue at small M: the prologue is ~3.3 us at T<=16; anything that lets the weight stream start before routing finishes (FAST1 pre-route issues exist as a flag: -DOKC_FAST1=) is worth one attempt if not already on.
5. Expert-order / unit scheduling at M=32: schedule the hottest expert first so the tail is short (unit-end spread 3-8 us at phase end).

Any change to the CUDA MoE source: compile is 100-200 s; use `./gate --stage compile` then `--stage correct`. Watch ptxas: a C7510/C7512/C7513 warning means wgmma got serialized and the kernel is 3-5x slower; treat it as a failure.

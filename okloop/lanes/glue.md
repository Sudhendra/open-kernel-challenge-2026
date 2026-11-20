# Lane: glue — launches, projections, boundaries, graph structure
Why: at T<=8 the layer is 9 kernels under PDL; each boundary costs ~1.2 us and each kernel ~3 us ramp/drain. d1s is 74 us vs a 44.6 us byte floor;
the glue (route 4, bnd1 3.6, bnd2 3.6, unpool 3.6) and the small projections (qkv 9, o 5.5, sk2f 19.5) are where the non-MoE, non-attention time is.
Also T=128/256 (verify16/32): nvjet GEMMs 26/21/16/8 us and 35/35/33/24 us, splitK 3.3 us.

Anchors in best.py:
- PDL globals ~90-91 (PDL, PDL_MAX_T), BFP_PDL_N ~9899 (dead: 2 is optimal), BFP_MAX_T ~10095, `bfp__ext = load_inline` ~9123 (projections: cgemv T=1, sk2 T<=32, sk2f fused pooler+router T<=16).
- Glue kernels: `_bnd1_apply` ~10377, `_route_a/_route_b` ~10396/10431, `_bnd2_sums/_bnd2_apply` ~10449/10470, `_route`/`_route_split` ~10517/10563; CUDA grafts `graft_glue_`, `graft_b2_` ~11817-11826, `sl_early_tail_` ~12681, `gle_o_` ~13602, `gle_b1x_` ~13891; GLE_* knobs ~13285-13290, 13819.
- Router FP32 at T<=8: `_router_fp32_gemv` ~13247/13270, `_pool_router_fp32` ~13722.
- kernel_fn ~13083 (last definition): the per-shape launch sequence. Count launches per shape first.

Dead list covers: sk2 R/CL/NS sweeps, sk2 T=16/32 restructuring, graph side branches, PDL depth, fused pooler+router at T=32, o+bnd1 fusion at T=8, custom wgmma GEMM at T=128/256, cuBLAS alternatives.

Ideas not on the dead list (pick ONE):
1. Launch count at T<=8: list every launch at d1s/d1f/v1s/v1f. Any Triton glue kernel that could be a tail of the preceding CUDA kernel (bnd2 into the unpooler cgemv epilogue; route_b into the pooler) is worth <= 4.8 us per shape; small but real on 4 of 12 shapes.
2. T=128/256 dense path: the nvjet picks are pinned; check the unpooler at T<=8 uses the padded [:1024] path and that T=128/256 do not pay a separate zero-fill or cast launch.
3. The `splitK 3.3 us` at v16s: what is it, and can its reduction fold into the consumer?
4. Residual boundary at T=128/256 (bnd 5.3+4.8 us at v16s, 8.4+6.7 at v32s): these scale with T, so they are bandwidth-bound passes over stream fp32 [T,8192]; one fused pass instead of sums+apply halves the traffic.
5. Graph-level: confirm PDL is on for every edge at T=128/256 (PDL_MAX_T=32 today; EXP-16 said 128 was a tie on the old base; re-check only the v32 shapes).

Everything here is small (0.2-0.8% per keep). That is the point: these stack.

Lead from the research scan (Nov 1, VkaZas on the guild board, H200): PDL trigger placement. Issuing griddepcontrol.launch_dependents
before the kernel's own wait costs 400-700 ns per boundary vs 830-1500 ns when triggered late. Audit where every kernel in the
chain triggers (projections, attention, MoE, glue) at d1s/d1f; moving triggers earlier where the dependent only needs the previous
kernel's outputs late is a few lines and applies to 9 boundaries per shape.

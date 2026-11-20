# Lane kdiet, 2026-11-19: patch byte diet (no default-path change)

Metric: `python3 harness/comment_trim.py lanes/kdiet --base 297e54fd --target-bytes 300000 --dry | grep "reduced diff"` (B = after full comment trim).
Baseline 01e47dbe25: B = 406,631.

| Commit | Cut | Files | B before -> after | Proof the default path is unchanged |
|---|---|---|---|---|
| 85ef802b61 | Kill switches removed, default arm kept: ALL_REDUCE_EARLY_PDL_TRIGGER (`make_cpp_args(.., pdl, pdl)`), ALL_REDUCE_WORKSPACE_INPUT/OUTPUT (condition term dropped), DFLASH_TP_CONTEXT_PROJECTION (`_project_context_tp` was always True: prewarm block dedented, `elif rows_replicated:`), DFLASH_CONTEXT_GEMM_PREWARM (`if is_cuda():`), FA3_VERIFY_SPLIT_POLICY, MIXED_CHUNK_FINAL_ONLY (`and any(...)`), MUSE_AG_NORM_ROW_KERNEL (`_row_kernel_disabled` + its `return False` gone), MUSE_DECODE_ROW_SPLIT, MUSE_PREFILL_KEY_SPLIT (`if tiles < 64`), MUSE_GROUPED_FP4_ROW_SPLIT (`_row_split` gone), MUSE_LAST_LAYER_READ_ROWS, MUSE_SPARSE_EXPERT_PUSH / EXPERT_PAIR_PUSH (`sparse_push=True, pair_push=True`), SHORT_PREFILL_DELAYER, SMALL_M_DECODE_PATHS (`_TUNED_32` unconditional in 17-32 rows; `min_fused_tokens=1` always). Tunables inlined at their defaults: HICACHE_QUIET_SYNC_INTERVAL=32, SHORT_PREFILL_DELAYER_MAX_TOKENS=4096 / MAX_DELAY_PASSES=4, DFLASH_PROMPT_LOOKUP_MIN_MATCH=8. DFLASH_BRANCH_LOG_INTERVAL (default 0 = off) removed with `_log_branch` and `_branch_steps` (the `_branch_stats` tensor is still allocated and passed to `branch_accept` as before). environ.py: whole entries deleted only. | all_reduce.py, muse_spark_all_gather_norm_residual.py, small_m_pdl_gemm.py, muse_spark_fp4_grouped.py, custom_all_reduce_v2.py, environ.py, flashattention_backend.py, muse_spark_bf16_moe.py, muse_spark_prefill_attention.py, scheduler.py, unified_radix_cache.py, muse_spark.py, dflash_worker_v2.py | 406,631 -> 399,695 (-6,936) | Every removed read was `EnvBool(False)` for a `DISABLE` name (or the tunable's declared default), so each condition reduces to its default arm by substitution; no fallback (shape/dtype/arch/try-except) was removed: e.g. AG-norm keeps `_supports_multicast_push`, decode-row split keeps `_kernel_built`/mode/len guards, `_row_split_mode` keeps its token-range guard, workspace view keeps all CA guards. `grep -rn` of every removed name over python/sglang, test/registered, docs, harness: no hits left (harness only sets SGLANG_DISABLE_MUSE_EXPERT_REPLICATION / ROUTER_STREAM_PRIORITY, not removed). |
| 9aebaeea05 | FA3 verify schedule: `_fa3_wave_fill`, `_write_splits_reference` and the `if not schedule.is_cuda` CPU branch of `write_splits` moved into test_fa3_verify_schedule.py (patched over `vs.write_splits` in setUp; exec'd in the TRITON_INTERPRET subprocess for the kernel parity test). | fa3_verify_schedule.py | 399,695 -> 397,877 (-1,818) | Production schedule buffers are CUDA (FA3 scheduling is registered for CUDA only); the Triton launch is byte-identical. Only callers of the reference were the CPU branch and the test. |
| 034639ae27 | Online placement tunables SGLANG_MUSE_ONLINE_PLACEMENT_{ROWS,ALPHA,MIN_GAIN,STAY_SLACK,LAYERS_PER_FORWARD} replaced by their defaults (60000, 0.5, 0.02, 0.02, 4); the env comments moved next to the constants. | muse_spark_online_placement.py, environ.py | 397,877 -> 396,581 (-1,296) | Same values; `max(1, 4 if layers_per_step is None else layers_per_step)` = old logic. |
| 4348429305 | SP overlap: `routed_ranks` and the non-CUDA tails of `ring_fold`, `routed_ring_fold`, `pair_fold`, `push_routed_rows`, `_store_routes` moved into test_muse_spark_sp_overlap.py (installed by setUpModule); a non-CUDA fold now raises TypeError. | muse_spark_sp_overlap.py | 396,581 -> 394,466 (-2,115) | muse_spark.py allocates every slot on `torch.device("cuda", ...)` (rows bf16, routes int32) and gates SP on `is_cuda()` + SM90, so the CPU tails were test-only; GPU branches untouched. `routed_ranks` had no other production caller (grep). Only test_muse_spark_sp_overlap.py exercises these paths. |

| 21f4ad3e26 | Test only: test_sp_region_skips_the_hidden_gather now uses an all-rows batch (> _READ_ROWS_MAX requests); it had relied on read rows being off. | test_muse_spark_unread_last_layer.py | 394,466 (no change) | test-only |

Total: 406,631 -> 394,466 (-12,165 B).

CPU suite (judge image, modal_cpu_stub): run 1 on 85ef802b61 tree: 5 failed, 742 passed, 24 skipped (4 = fp4 narrow-ring tests patching removed `_row_split`, fixed in 9aebaeea05; 1 = SP gather test, fixed in 21f4ad3e26). Run 2 on 21f4ad3e26 (out/kdiet_cpu_2.log): 746 passed, 0 failed, 24 skipped, 3343 subtests passed. Baseline 753 passed; the 7 fewer are exactly the 7 deleted kill-switch tests (ag_norm x2, fa3 schedule, mixed-chunk, short-prefill delayer, hicache quiet sync, fp4 narrow ring). The `CustomAllReduceV2 ... no attribute 'disabled'` __del__ warning in the log is pre-existing (also in kfinal_cpu_2 / kdatt_cpu_suite1).

Tests removed or changed (they tested a removed off-switch):
- test_muse_spark_ag_norm_row_policy: test_kill_switch, test_kill_switch_does_not_apply (deleted).
- test_small_m_pdl_gemm_build: the override(True) block of the table test (deleted).
- test_fa3_verify_schedule: test_kill_switch_declared_default_on (deleted).
- test_muse_spark_all_reduce_view: "switch" subcase and the INPUT-switch half of the room test (renamed test_input_view_needs_room).
- test_muse_spark_prefill_attention: key-split override replaced by `patch.object(gate, "_key_splits", return_value=1)`; test_kill_switch_and_unbuilt_kernel -> test_unbuilt_kernel.
- test_muse_spark_sp_overlap: two `assertFalse(envs.SGLANG_DISABLE_..._PUSH.get())` lines.
- test_muse_spark_unread_last_layer: the "without the opt-in" half of test_read_batch_is_the_stock_layer_bit_for_bit.
- test_scheduler_mixed_chunk_final_only: test_kill_switch_mixes_every_chunk (deleted).
- test_scheduler_short_prefill_delayer: test_kill_switch (deleted).
- test_hicache_quiet_sync: test_default_interval_comes_from_the_env (deleted).
- test_muse_spark_fp4_grouped_narrow_ring: test_kill_switch (deleted); `_row_split` patch dropped.
- test_muse_spark_online_placement: test_stay_slack_from_env_reaches_the_fit -> test_stay_slack_reaches_the_fit (env override half dropped); the rows/min-gain test reads the constants.
- test_dflash_context_gemm_prewarm: dropped the `_project_context_tp = True` setup line.

Skipped: SGLANG_SPEC_TP_SYNC (stock variable; default changed to "init", a mechanism not a switch); SGLANG_DFLASH_BRANCH / _MIN_GAIN / SGLANG_ENABLE_DFLASH_PROMPT_LOOKUP (kept by instruction); startup banners (harness/judge reads may grep them; not needed for the target); `_branch_stats` (would change the branch_accept launch).

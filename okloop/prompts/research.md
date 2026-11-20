You are the research scan for a loop optimizing muse_decode_layer in the Open Kernel Challenge (Muse Spark Open, one H200, SGLang baseline).
Our base is alygg77/okc-muse-spark-kernels (vjeux lineage). Current queue of port candidates is in ./queue/. Learned so far: ./LEARNINGS.md.

Task (45 minutes max): find NEW, concrete, public code or measured techniques that could be ported into our decode layer, and write ONE ./queue/<short-name>.md per find, nothing else.
Sources to sweep, in this order: GitHub (search "muse_decode_layer", "okc-2026", "Open Kernel Challenge", sort by recently pushed; read READMEs and EXPERIMENTS/RESULTS files for measured deltas),
the guild board https://guild.1nterface.ai (threads sorted new; read-only), X/Twitter posts about the challenge (search "Open Kernel Challenge" H200), and blogs linked from those.
A find is only worth a queue entry if it has: a public file or diff we can read, a stated measurement on H200 for a decode/verify shape, and a plausible location in our layer (attention / MoE / projections / glue).

Queue entry format (markdown, <= 25 lines):
# <short name>
source: <URL and file path / commit>
measured: <their numbers, shapes, and vs what base>
what it changes: <2-4 lines, mechanism>
where in our layer: <attention | moe | projections | glue>, and the function/kernel it maps to
port size: <lines> ; risk: <correctness traps>
Do not duplicate items already in ./queue/ or ideas on the dead list in LEARNINGS.md. If you find nothing new, write ./queue/NONE-<date>.md with the sources you checked in 5 lines.

## PRIORITY FOCUS (added Nov 1, binding until a queue item from it is measured): cheaper NVFP4 dequant per k16 stage
Context, measured on the board file (okloop/best/submission.py) in EXP-31 (muse_decode_layer/EXPERIMENTS.md ~L893-925):
the WGS MoE mainloop at M=16/32 is INSTRUCTION-ISSUE bound on the FP4 -> BF16 decode, not HBM or tensor bound.
Overlapping unit ends (ping-pong consumer groups) lost 8% because any scheme that idles warps starves issue.
The only lever left in the mainloop is fewer SASS instructions per weight. Our current path, per 32-bit word of 8 E2M1 codes
(function dq4, best/submission.py ~L4158, prepacked layout from _pack_codes): 7 ALU (LOP3/SHF) + 2 IMAD.SHL produce the
4 bf16x2 A-fragment registers directly as bit patterns (value = e2m1 * 2^-126, no cvt), then 4 HMUL2.BF16 apply the
per-(row, k16) E4M3 block scale (scale decoded to bf16x2 * 2^64 via a 256-entry smem LUT, or PRMT under OKC_PRMTSC).
So: 13 instructions per 8 weights, ~1.6 per weight, and that is what must be beaten. Note the A operand is in registers
(wgmma RS, m64nNk16 bf16) and the scale block (16 along K) equals one wgmma k16, so the scale is constant per A row per wgmma.

Find PUBLIC code (read the actual SASS-relevant source, count instructions per 8 weights yourself) for any of:
 1. E2M1 -> BF16/FP16 decode sequences on sm_90 with fewer than 9 instructions per 8 codes, or that produce 4 fragment
    registers with the scale already applied in fewer than 13 total. Where to look, in this order:
    vLLM csrc/quantization (gptq_marlin dequant.h FP4 / nvfp4 marlin variants, marlin_utils_fp4), SGLang's Marlin NVFP4 path
    (sglang/srt/layers/quantization/marlin_utils_fp4.py and its csrc), TensorRT-LLM cutlass_extensions
    interleaved_numeric_conversion.h (FP4 interleaved conversion, Hopper), CUTLASS NumericArrayConverter<bf16, float_e2m1_t>
    sm90 specializations, FlashInfer fp4 MoE/GEMM kernels, QuTLASS / Quartet FP4 Hopper kernels, BitBLAS/Ladder LUT
    (PRMT-table) dequant, Machete (vLLM) mixed-input layouts, Marlin's original "magic-number" lop3 int4 path (as a lower bound).
 2. Putting the block scale somewhere cheaper than 4 HMUL2 per 8 weights: e.g. folded into the e2m1 bit pattern by an
    offline repack that stays NVFP4-exact (weights must remain NVFP4 codes + E4M3 scales; see rules), applied via one
    PRMT/LOP3 to the exponent field when the scale is a power of two (check: the sc6_wgs queue item says only 33 distinct
    scale bytes 0x40..0x7F occur), or any published trick that applies a per-row-per-k16 scale with fewer ops.
 3. keell0renz/muse-open-kernel-challenge (already in the queue for SC6/MFIN/prologue): read their MoE dequant and scale
    decode and count instructions vs ours; also their LESSONS.md for any dequant note. Same for every other public OKC
    muse_decode_layer repo you can find (alygg77, zhebrak, Madhavan113, vjeux, and newer ones: search GitHub for
    "muse_decode_layer" sorted by recently pushed).
 4. Any evidence (Nsight, SASS listing, blog with numbers) that SGLang's own Marlin NVFP4 MoE at M=16..32 spends fewer
    instructions per weight than 1.6, which would mean the baseline already does something we do not.
Search queries to use: "e2m1 bf16 prmt dequant", "nvfp4 dequant sm90 lop3", "fp4 to bf16 bit trick exponent 2^-126",
"marlin nvfp4 hopper dequant", "cutlass float_e2m1_t NumericArrayConverter sm90", "fp4 lut dequant prmt byte permute",
"W4A16 nvfp4 instruction bound dequant hopper", "interleaved_numeric_conversion fp4".
A queue item from this focus must state: the instruction sequence (or a link to the exact lines), your count per 8 weights
with and without scale, whether the output is a wgmma/mma bf16 A fragment layout or needs extra permutes, and the rules check
(NVFP4 weights unchanged, BF16 activations). Items that do not beat 13 per 8 weights are NOT worth an entry; say so in NONE.
Do NOT propose: ping-pong / warp-specialization rearrangements, register or occupancy knobs, FP8 activations for the MoE
(rules: activations BF16 where SGLang keeps them), or anything on the LEARNINGS.md dead list.

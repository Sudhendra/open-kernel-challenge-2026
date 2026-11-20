# Kernel problems

The kernel half of the challenge (Oct 22 to Nov 4, 2026) had five single-GPU problems, each one piece of a Muse Spark Open decode or prefill step on an H200. It carried no prizes; its overall board set the queue priority for the serving track.

Each file below is the exact source of a scored job. `python3 kernels/verify.py` recomputes the judge's content hash for each and compares it with the job's.

| Problem | File | Scored job | Speedup vs SGLang | Board rank (read Nov 20) |
|---|---|---|---:|---:|
| Gated attention decode | [muse_attention_decode.py](muse_attention_decode.py) | `49f87f365df1496e` | 2.630x | 252 |
| Decoder layer | [muse_decode_layer.py](muse_decode_layer.py) | `8178ab1226944045` | 2.316x | 85 |
| Routed experts (NVFP4 W4A16 MoE) | [muse_moe_w4a16.py](muse_moe_w4a16.py) | `4a34800e1b664536` | 2.394x | see below |
| Prefill layer | [muse_prefill_layer.py](muse_prefill_layer.py) | `ea1c10521bde4fb5` | 1.898x | 146 |

Overall kernel board: rank 85.

Two gaps, stated plainly:

- **Routed experts.** The board entry is a later job (`dc35799d56f14c1e`, 2.418x, rank 153). Its source is not in this workspace, so the file here is the earlier scored one.
- **Dense projections (`muse_bf16_proj`).** An entry scored 1.567x during the track. It is no longer on that board and is not published here.

## Whose work this is

These files are not original kernels. Every submission to the challenge is Apache-2.0 licensed, many participants published theirs, and the leading entries were built by stacking each other's work. Each file here is a stack of other participants' public submissions (vjeux, alygg77, team keell0renz, ZhimaoL, chenwainuo and others) that was ported, re-validated and in places extended with small changes. The comment block at the top of each file lists its lineage line by line, and the original authors' attribution headers are kept. See [NOTICE](../NOTICE).

The original work in this repository is the serving patch in [serving/](../serving/).

## Running one

The task files (`reference.py`, `checks.py`, `baseline.py`) come from the challenge CLI and are not copied here:

```
okc init muse_decode_layer work/
cp kernels/muse_decode_layer.py work/submission.py
```

The files need an H200 or H100 (`sm_90a`) and the challenge's SGLang image.

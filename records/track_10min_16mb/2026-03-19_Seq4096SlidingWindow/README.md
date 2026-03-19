# Training Opt Seq4096 + Sliding Window Eval (stride=64)

This record combines two independent improvements over the naive baseline:

1. **Longer training context (seq_len=4096) with aggressively tuned Muon optimizer**
2. **Sliding window evaluation (stride=64) for near-full context on every scored token**

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Sequence length: `TRAIN_SEQ_LEN=4096`
- Batching: `TRAIN_BATCH_TOKENS=393216` (3/4 batch for more optimizer updates per second)
- Learning rates: `TIED_EMBED_LR=0.030 MATRIX_LR=0.020 SCALAR_LR=0.020`
- Muon optimizer: `MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_MOMENTUM_WARMUP_START=0.92`
- Schedule: `WARMDOWN_ITERS=3000`
- Eval: `EVAL_STRIDE=64` (sliding window, each scored token gets 4032 tokens of context)

## Command

```bash
RUN_ID=combined_seq4096_sw64 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Timed training stopped at `11478/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0071`, `val_bpb:1.1887`
- Post-quant roundtrip eval (standard): `val_loss:2.0142`, `val_bpb:1.1929`
- **Post-quant sliding window eval: `val_loss:1.9937`, `val_bpb:1.1808`**
- Exact printed metric: `final_sliding_window_eval_exact stride:64 val_loss:1.99374987 val_bpb:1.18081285`
- Train time: `599949ms` (`step_avg:52.27ms`)
- Sliding window eval time: `278940ms` (under 10-minute eval budget)
- Peak memory: `7655 MiB allocated`, `8156 MiB reserved`
- Serialized model int8+zlib: `15824445 bytes`
- Code size: `54433 bytes`
- Total submission size int8+zlib: `15878878 bytes`

## Approach

This submission stacks three orthogonal improvements:

### 1. Longer training context (seq_len=4096)
Each training sequence sees 4x more context than the 1024-token baseline, giving the autoregressive model much better signal per token. This costs ~52ms/step (vs ~43ms at seq_len=1024), but the quality improvement far outweighs the fewer total steps (11,478 vs ~13,780).

### 2. Aggressive Muon optimizer tuning
- **Higher momentum (0.99 vs 0.95):** Stronger gradient smoothing for better convergence.
- **Lower learning rates (0.020 vs 0.04):** Dramatically reduces int8 quantization loss (0.004 BPB quant penalty vs 0.007+ at default LR) while maintaining similar pre-quant quality.
- **3/4 batch (393K vs 524K tokens):** More optimizer updates per wallclock second.
- **Extended momentum warmup (1500 steps from 0.92):** Prevents early instability with the higher momentum.
- **Longer warmdown (3000 steps):** Proportionally longer LR decay for the ~11,500-step run.

### 3. Sliding window evaluation (stride=64)
Standard evaluation chops the validation set into non-overlapping 4096-token blocks, meaning the first tokens in each block get minimal context. Sliding window evaluation shifts the window by only 64 tokens at a time, scoring only the last 64 tokens per window. Each scored token sees 4032 tokens of prior context, dramatically reducing the "cold start" penalty. This improves BPB by ~0.012 over standard post-quant eval (1.1808 vs 1.1929).

The evaluation uses a `forward_logits()` method that returns logits without computing loss, allowing us to score only the relevant trailing tokens per window. The sliding window runs on 8 GPUs in parallel with batched windows (32 per forward pass) and completes in ~279 seconds.

## Training Volume

- Global batch: `393216` tokens/step
- Total train tokens seen: `~4.5B` (11,478 steps × 393,216 tokens)
- Dataset: 20 shards of fineweb10B_sp1024

## Included Files

- `train_gpt.py` (standalone script with all changes baked in)
- `README.md` (this file)
- `submission.json` (leaderboard metadata)
- `train.log` (full training log from the canonical run)

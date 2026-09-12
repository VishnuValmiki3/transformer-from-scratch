# Benchmarks

All numbers come from a single unattended run on one Kaggle **Tesla T4** (free tier),
reproducible with [`notebooks/kaggle_end_to_end.ipynb`](../notebooks/kaggle_end_to_end.ipynb).
Raw logs are committed alongside this file.

## Setup

| | |
|---|---|
| GPU | Tesla T4 (16 GB), fp16 autocast |
| Model | 17,576,448 parameters — 4 layers, d_model 512, 8 heads, d_ff 1344, context 256 |
| Tokenizer | byte-level BPE, vocab 10,000 (256 bytes + 1 special + 9,743 merges) |
| Corpus | TinyStories V2 (GPT-4), 72,950,901 train / 5,468,834 validation tokens |
| Optimizer | AdamW (from scratch), cosine schedule with 500-step warmup, grad clip 1.0 |

## Data pipeline

| Stage | Result |
|---|---|
| BPE training (30 MB sample) | 9,743 merges in **30.8 s** |
| Corpus encoding (4 workers) | **1,583,798 tokens/s**, 4.11 bytes/token |
| Compression | 300 MB of text → 73M tokens |

BPE merges are learned from a 30 MB sample rather than the full corpus: merge statistics
saturate early, and this keeps a pure-Python trainer tractable. Encoding splits the corpus
on `<|endoftext|>`, the only boundary guaranteed to tokenize identically to the un-split
text, which also makes the parallel path byte-identical to the sequential one.

## Pretraining

10,000 steps × 32 batch × 256 context ≈ 82M tokens, in **35 min 12 s**.

| Metric | Value |
|---|---|
| Median throughput | **40,838 tokens/s** |
| Final train loss | 1.5234 |
| Best validation loss | **1.5519** |
| Validation perplexity | **4.72** |

| Step | Val loss | Perplexity |
|---|---|---|
| 250 | 3.4840 | 32.59 |
| 500 | 2.7629 | 15.85 |
| 1,000 | 2.2737 | 9.72 |
| 2,000 | 1.9648 | 7.13 |
| 5,000 | 1.7013 | 5.48 |
| 10,000 | **1.5519** | **4.72** |

![training curve](training_curve.png)

## Ablation: embedding initialization under weight tying

The first full run initialized the embedding at std = 1.0 and started training at a loss of
**342**, against the ln(10,000) = 9.21 a uniform predictor scores. Weight tying makes the
embedding matrix double as the output projection, so logits are a `d_model`-long dot product
against it: logit std lands near √512 ≈ 22.6, which is exactly what was measured (22.59).

Re-running with GPT-2's std = 0.02, changing nothing else:

| Step | std = 1.0 | std = 0.02 |
|---|---|---|
| 10 (initialization) | 342.50 | **9.19** |
| 250 | 8.95 | 3.48 |
| 500 | 4.47 | 2.76 |
| 1,000 | 3.16 | 2.27 |
| 2,000 | 2.51 | 1.96 |
| 5,000 | 2.02 | 1.70 |
| **10,000 (val loss)** | 1.8029 | **1.5519** |
| **Perplexity** | 6.07 | **4.72** |

A one-line change, **22% lower final perplexity**. The damage is concentrated early — at
step 250 the mis-initialized model was still scoring 8.95, barely better than guessing, having
spent that entire budget undoing its own initialization. RMSNorm normalizes the input path
immediately, so the small init costs nothing on the way in.

Raw logs: [`metrics_init_std1.0.jsonl`](metrics_init_std1.0.jsonl) and
[`metrics_init_std0.02.jsonl`](metrics_init_std0.02.jsonl).

## LoRA fine-tuning

Rank 8, α = 16, applied to `q_proj` and `v_proj` across all 4 layers (8 modules), 1,000 steps
on tiny-shakespeare in **1 min 44 s**.

| Metric | Value |
|---|---|
| Trainable parameters | **65,536 of 17,641,984 (0.37%)** |
| Adapter file size | **261 KB** (vs 201 MB for the full checkpoint) |
| Style validation loss | 4.5872 (perplexity 98.22) |

### What the adapter bought, and what it cost

| Domain | Base | + LoRA | Change |
|---|---|---|---|
| Shakespeare (adaptation target) | 824.85 | **96.06** | **−88.4%** |
| TinyStories (original domain) | 4.73 | 53.71 | **+1034.7%** |

The second row is the one usually left out. Adapting hard to a small, stylistically distant
corpus causes severe forgetting of the original domain — an 88% perplexity win on the target
bought at more than a 10× loss on the source. With 0.37% of parameters trainable, that is not
a capacity limitation: it is the adapter rotating attention toward the new distribution.
Mitigations worth trying are a lower learning rate, fewer steps, or mixing source data into
the fine-tuning set; none were applied here, so the trade-off is visible rather than hidden.

## Sample output

Base model, prompt `"Once upon a time"`:

> Once upon a time, there was a little girl named Lily. She had a big, soft teddy bear named
> Mr. Bear. Lily and Mr. Bear loved to play together all day long.
>
> One sunny day, Lily and Mr. Bear went to the park to play. They saw a pretty tree with lots
> of leaves. Lily said, "Mr. Bear, let's play hide and seek!" Mr. Bear nodded and they started
> to play.

Same checkpoint and prompt with the adapter loaded:

> Once upon a time,
> So F image had a bigXidy and s dip
> T mentioned the light of a princess.
>
> **TLuckily:**
> I will must say her size, to Bidtam,
> I will have a nation of a King.
>
> **MANGIONIO:**
> Just, then, in nothing if it will not be
> In our message to Min'se journey of a land

The interesting part is *what* transferred. The adapter picked up Shakespeare's **structure** —
capitalized speaker labels, colons, verse line breaks, dialogue framing — while the
**vocabulary stays TinyStories** (princess, message, journey). That is what adapting only
`q_proj`/`v_proj` should do: it reshapes what attends to what, while the embedding and
feed-forward layers holding lexical knowledge stay frozen. Token-level quality is poor, which
is expected from a 17M-parameter model given 1,000 steps on 393K tokens.

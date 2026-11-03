# Round 2 fine-tuning: findings (post-deadline follow-up)

**Not part of the hackathon submission.** The submitted results live on `main` (`results/`). Round 2 follows the
next steps in the README draft: select checkpoints on validation and drift data rather than the test set; record
the loss curve and LoRA settings; run the text-only (no-metadata) ablation for Nemotron Super; re-check
reproducibility. Full tables: [`summary.md`](summary.md). Loss curves: [`loss_curve.png`](loss_curve.png).

## Setup

- **Base model:** `nvidia-nemotron-3p5-super-vl-120b-a12b-bf16`.
- **Training:** LoRA rank 16, alpha 32, adapters on MLP, attention and unembedding (SDK defaults). Constant learning rate 1e-4, batch 32, seed 42, loss on the answer tokens only. Trained on Fireworks Serverless Training with the same frozen 2,400 training examples as round 1.
- **Selection:** on dev data only. That means the 200-case validation split plus two new dev slices built by `scripts/build_dev.py`: 200 later-corpus (2008) emails and 169 recent synthetic scams, disjoint from every frozen test file. Every epoch is still reported on the frozen tests.
- **Run A (`meta-r16`):** metadata prompt, 3 epochs, scored after every epoch.
- **Run B (`text-r16`):** text-only prompt, 1 epoch.
- **Spend:** about $10.40 for round 2 (training $8.61, in-session sampling $1.79). The whole project is about $18.70 of the $50 credit.

## Results on the frozen test sets

| Model | Test acc (600) | Errors | FPR | vs Kimi K3, paired | Later corpus (200) | Recent synthetic (200) |
|---|---|---|---|---|---|---|
| **Text-only, epoch 1 (round 2)** | **98.0%** | 12 | 0.7% | **ΔF1 +1.9 [+0.5, +3.3], p = 0.019** | 85.5% | **90.5%** |
| Metadata, epoch 1 (round 2) | 98.0% | 12 | 1.3% | n/a | 86.0% | 86.0% |
| Metadata, epoch 2 (round 2) | 97.3% | 16 | 3.3% | n/a | **91.5%** | 62.5% |
| Metadata, epoch 3 (round 2, dev-selected) | 97.7% | 14 | 0.3% | ΔF1 +1.5 [+0.1, +3.0], p = 0.064 | 86.5% | 86.5% |
| Metadata, epoch 1 (round 1, submitted) | 97.8% | 13 | 1.0% | ΔF1 +1.7 [+0.2, +3.3], p = 0.041 | 85.5% | 86.0% |
| Kimi K3 (frontier) | 96.2% | 23 | 2.7% | n/a | 92.0% | 94.0% |

## What we learned

1. **The round-1 result reproduces.** The round-2 metadata epoch-1 adapter gets 98.0% vs 97.8% in round 1. Paired on the same 600 cases, they differ on 5 cases (p = 1). This also settles the checkpoint provenance: the submitted numbers are epoch 1.
2. **More epochs did not help, and epoch 2 was unstable.**
   - In-distribution accuracy stays flat across epochs (98.0, then 97.3, then 97.7).
   - Epoch 2 traded robustness for email drift: later-corpus accuracy rose to 91.5% while recent-synthetic accuracy collapsed to 62.5%. Epoch 3 recovered.
   - Validation alone would not have shown this. The two drift dev slices predicted it (91.0% and 58.0% on dev).
   - The rule picked epoch 3 by 0.1 point of dev mean over epoch 1, which is within noise. One epoch is the better buy at a third of the cost.
3. **Metadata is not needed once the model is fine-tuned.** At epoch 1, text-only and metadata are identical in distribution and on later-corpus email (paired p = 1 on both). On recent synthetic scams text-only is better: **90.5% vs 86.0%**, with 9 cases only text-only got right and 0 the other way (McNemar **p = 0.004**). The fine-tune seems to over-rely on the metadata block, which is sparse on new-style scams. Metadata still helps the untuned frontier prompts (DeepSeek +1.1 F1 in the round-1 ablation).
4. **Best model so far: the text-only epoch-1 fine-tune.** It has the strongest paired win over Kimi K3 (p = 0.019) and halves the out-of-distribution gap to Kimi K3 (from 8 points to 3.5). The later-corpus gap (85.5% vs 92.0%) remains.

## Next steps

- Combine what epoch 2 learned about email drift with epoch-1 robustness: a decaying learning rate or warmup, early stopping on the dev mean, and a mix of recent email and hard legitimate look-alikes in training. The look-alikes must come from sources other than the test corpora.
- Re-run the text-only recipe on a second seed and a different split before generalizing the Kimi K3 comparison.
- Deploy the chosen adapter to measure real serving latency and cost. Adapters sampled in a training session disappear when the session ends.

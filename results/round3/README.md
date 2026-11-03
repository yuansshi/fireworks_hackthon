# Round 3 fine-tuning: findings (post-deadline follow-up)

**Not part of the hackathon submission** (which is `main`). Round 3 tests the round-2 next steps. All runs are text-only, which was round 2's best prompt, on NVIDIA Nemotron 3.5 Super (120B-A12B) with LoRA rank 16 and alpha 32 on Fireworks Serverless Training. Checkpoints are selected **on dev only** (validation plus the later-corpus and recent-synthetic dev slices), and every checkpoint is reported on the frozen tests. Tables: [`summary.md`](summary.md). Loss curves: [`loss_curve.png`](loss_curve.png). Spend: about $14.40 for round 3, so about $33.10 of the $50 credit in total.

| Run | Recipe |
|---|---|
| A `text-cos` | 2 epochs; peak learning rate 1e-4 with 10% warmup, then cosine decay to 1e-5; a checkpoint every half epoch |
| B `text-cos-extra` | A, plus 600 extra training rows from `AreLit/PhishNChips`: 300 recent-style phishing, 150 legitimate, and 150 hard legitimate look-alikes whose link domain differs from the sender's. The synthetic content is MIT-licensed; its embedded URL indicators are cleared for academic research with attribution. None of the test sets use this source |
| C `text-seed7` | Round 2's best recipe (1 epoch, constant learning rate) with a second seed (7) |

## Results (selected checkpoint, frozen tests)

| Model | Test acc (600) | Errors | FPR | vs Kimi K3, paired | Later corpus | Recent synthetic |
|---|---|---|---|---|---|---|
| A: decaying learning rate, epoch 1.0 | 98.0% | 12 | 1.0% | ΔF1 +1.9 [+0.3, +3.6], p = 0.027 | 84.5% | 86.5% |
| B: A + extra data, epoch 1.5 | **98.3%** | **10** | 0.7% | **ΔF1 +2.2 [+0.8, +3.9], p = 0.004** | 86.0% | 82.5% |
| C: round-2 recipe, seed 7, epoch 1 | 97.7% | 14 | 0.7% | ΔF1 +1.5 [−0.0, +3.2], p = 0.078 | 85.5% | **92.5%** |
| Round 2 best: text-only, seed 42, epoch 1 | 98.0% | 12 | 0.7% | ΔF1 +1.9 [+0.5, +3.3], p = 0.019 | 85.5% | 90.5% |
| Kimi K3 (frontier) | 96.2% | 23 | 2.7% | n/a | 92.0% | 94.0% |

## What we learned

1. **A decaying learning rate makes longer training safer, but not better.**
   - The out-of-distribution collapse in round 2 (62.5% at epoch 2 with a constant learning rate) shrinks to a gradual slide: 86.5%, then 83.5%, then 81.0% across epochs 1.0, 1.5 and 2.0.
   - In-distribution accuracy keeps creeping up over the same span (98.0%, then 98.3%, then 98.5%).
   - The dev slices caught this trade and early stopping picked epoch 1.0. Selecting on the in-distribution test would have chosen epoch 2.0 and quietly lost about 5 points out of distribution.
   - Decay did **not** reproduce round 2's epoch-2 jump on later-corpus email (91.5%). That gain came with the high-learning-rate memorization phase that also wrecked robustness.
2. **The extra recent emails and hard negatives did not close the later-corpus gap.**
   - Later-corpus accuracy stayed at 85–86%, against 92.0% for Kimi K3, and recent-synthetic accuracy fell to 82.5%.
   - The 2008 test corpus is mostly bulk spam, while PhishNChips is targeted, LLM-written phishing, so it teaches a different style.
   - The extra data did give the strongest in-distribution result and the clearest win over Kimi K3 (10 errors vs 23, p = 0.004). That is useful, but it wasn't what the data was for.
3. **The text-only, one-epoch recipe replicates across seeds.**
   - Seeds 42 and 7 give 98.0% and 97.7% on the test set, and 90.5% and 92.5% on recent synthetic scams.
   - Out-of-distribution scores move about ±3 points between runs of the same recipe, so treat differences under ~5 points on the 200-case sets as noise.
4. **The win over Kimi K3 in distribution is consistent, but modest.**
   - Across the four text-only runs, the selected checkpoint beats Kimi K3 by ΔF1 +1.5 to +2.2 points every time.
   - It is significant at p < 0.05 in 3 of the 4 runs, and the seed-7 run gives p = 0.078.
   - Say "consistently ahead on this test set", not "always significantly better".

## Recommendation

- **Default recipe: text-only, 1 epoch, constant learning rate 1e-4** (round 2's best). It gives the best balance, about 98% on the test set and 90–92% on recent synthetic scams, and it is the cheapest at about $2 of training.
- **If in-distribution precision matters most:** use run B's recipe (98.3%, FPR 0.7%), accepting weaker out-of-distribution robustness.
- **The later-corpus gap needs different data or a fallback, not more tuning.**
  - Data: later-year bulk-spam email, from a source other than the test corpus.
  - Or route low-confidence and out-of-distribution traffic to a frontier fallback.

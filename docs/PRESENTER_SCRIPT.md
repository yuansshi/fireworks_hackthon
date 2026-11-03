# Video manuscript: read it yourself (target 1:45–1:50)

A script to read aloud over `docs/slides.pdf`. It follows the required order: **The Problem**, then **Tech Stack**,
then **Live Demo + Code**, then the **before/after fine-tuning** result. **Bold** = stress it. Times assume a relaxed
150 words per minute; the whole script is **259 words**, about **104 s**, leaving room for pauses under the 2:00 limit.

**Pronunciation:** Nemotron = *NEM-uh-tron* · Kimi K3 = *KEE-mee K-three* · LoRA = *LOR-uh*.

---

### Slide 1 · The problem: Title  (0:00–0:06)

> This is **PhishGuard**: an NVIDIA Nemotron model we fine-tuned on Fireworks to catch scams on every channel.

*Cue:* Look at the camera or slide; calm start. Then advance.

### Slide 2 · The problem: Problem and user  (0:07–0:22)

> Scams hit email, texts, links and phone calls, and fraud teams must screen all of it before customers act. Frontier models are accurate, but at **sixteen hundred dollars per million messages**, they're too expensive to run on everything.

*Cue:* Point to **$1,600** when you say it. Then advance.

### Slide 3 · Tech stack: What the stack made possible  (0:23–0:36)

> Our stack: **Fireworks Serverless Training** fine-tuned a **hundred-and-twenty-billion-parameter Nemotron** with LoRA in **eleven minutes**, for about **two dollars**, with no GPUs and no deployment. Fireworks inference ran the frontier baselines through the same API.

*Cue:* Move down the four rows as you name them. Then advance.

### Slide 4 · Tech stack: The evaluation  (0:37–0:44)

> We froze one evaluation: **twenty-four hundred** training examples and **six hundred held-out cases**, scored identically for every system.

*Cue:* Point to 2,400 and 600. Then advance.

### Slide 5 · Live demo: Animated terminal run  (0:45–0:57)

> Here it is on held-out messages: our verdict next to **Kimi K3**, with the cost per million messages. Same answers, a fraction of the price, and both miss one tricky link.

*Cue:* Walk down the four examples top to bottom; point at the cost column. Then advance.

### Slide 6 · Code: The training loop  (0:58–1:08)

> The training code is a short loop: a LoRA client, loss only on the answer tokens, an optimizer step, and a checkpoint we sample directly.

*Cue:* Point at forward_backward and optim_step. Then advance.

### Slide 7 · Code: Features and scoring  (1:08–1:18)

> Each message gets cheap metadata like urgency words and suspicious links, and every model is scored on the same cases with a paired test.

*Cue:* Point at the metadata lines, then the McNemar lines. Then advance.

### Slide 8 · Results: before vs after: Before vs after fine-tuning  (1:19–1:34)

> Now the key result. **Before** fine-tuning, Nemotron scored **ninety point seven percent**. **After**: **ninety-seven point eight**. Errors fell from **fifty-six to thirteen**, and phone-call false alarms from **thirty-four percent to two**. One regression: slightly worse on newer, unseen email.

*Cue:* **Slow down here.** Point at 90.7% → 97.8%, then the table. Then advance.

### Slide 9 · Results: vs the frontier: Beats the frontier  (1:35–1:43)

> That beats the frontier: **thirteen errors versus twenty-three** for Kimi K3, statistically significant, at about **one seventh of the cost**.

*Cue:* Point at 13 vs 23. Then advance.

### Slide 10 · Close: Why adopt  (1:43–1:49)

> PhishGuard: frontier-beating accuracy from an open NVIDIA model you can run yourself. Thanks!

*Cue:* Smile; stop recording two seconds after. Then advance.

---

## 60-second version (if you want it shorter)

Slides 1 → 3 → 8 → 9 → 10.

> This is PhishGuard: an NVIDIA Nemotron model fine-tuned on Fireworks to catch scams in email, texts, links and calls. Frontier models are accurate, but cost sixteen hundred dollars per million messages. With Fireworks Serverless Training we fine-tuned a hundred-and-twenty-billion-parameter Nemotron with LoRA in eleven minutes, for about two dollars, with no GPUs. On six hundred held-out cases, fine-tuning took accuracy from **ninety point seven to ninety-seven point eight percent**, cut errors from **fifty-six to thirteen**, and dropped phone-call false alarms from **thirty-four percent to two**. That beats Kimi K3 at about one seventh of the cost. The tradeoff: it's slightly weaker on newer, unseen email, so we'd refresh it with new data. Everything is reproducible in our repo.

## How to record (about 10 minutes)

1. Open `docs/slides.pdf` in Preview or Keynote, full screen.
2. Press **Cmd+Shift+5**, choose *Record Entire Screen*, set *Options → Microphone* to your mic, then click *Record*.
3. Read each block, advancing at each cue. Pause about half a second after each slide change.
4. Stop with the menu-bar stop button. In QuickTime use *Edit → Trim* to cut the start and end, keep it **under 2:00**, then *File → Export As → 1080p*.
5. Do one rehearsal with a timer first. If you run long, drop the Slide 7 sentence (the scoring code is also visible on screen).

Every number here matches `results/summary.md` (600 identical held-out cases).

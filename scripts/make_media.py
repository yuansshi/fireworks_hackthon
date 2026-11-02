"""Build the pitch deck (PNG + PDF) and the narrated 2-minute video from measured results.

    uv run python scripts/make_media.py            # slides + narration + video
    uv run python scripts/make_media.py --slides   # slides only

Outputs: docs/slides/slide_NN.png, docs/slides.pdf, docs/VIDEO_SCRIPT.md, docs/video/phishguard_2min.mp4
Needs: Google Chrome (headless render), macOS `say` (narration), ffmpeg/ffprobe.
Every number on a slide comes from results/summary.md (test set: 600 identical held-out cases).
"""

from __future__ import annotations

import argparse
import html
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SLIDES = DOCS / "slides"
BUILD = ROOT / "results" / "_scratch" / "media"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VOICE, RATE = "Samantha", 182
REPO = "github.com/yuansshi/fireworks_hackthon"

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { width: 1920px; height: 1080px; background: #fbfbf8; color: #111; overflow: hidden;
       font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; position: relative; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; width: 22px; background: #76b900; }
.wrap { position: absolute; left: 130px; right: 130px; top: 96px; bottom: 90px; }
.kicker { font-size: 30px; color: #4a4a46; letter-spacing: .5px; margin-bottom: 26px; }
h1 { font-size: 66px; line-height: 1.12; font-weight: 750; letter-spacing: -.5px; }
h2 { font-size: 60px; font-weight: 750; letter-spacing: -.4px; }
.rule { height: 5px; background: #76b900; width: 100%; margin: 30px 0 46px; border-radius: 3px; }
.sub { font-size: 36px; color: #3d3d3a; line-height: 1.4; }
.foot { position: absolute; left: 130px; right: 130px; bottom: 44px; font-size: 25px; color: #6b6b66;
        display: flex; justify-content: space-between; }
.stats { display: flex; gap: 0; margin-top: 10px; }
.stat { flex: 1; padding: 6px 48px 0 0; border-right: 3px solid #e2e2dc; margin-right: 48px; }
.stat:last-child { border-right: none; margin-right: 0; }
.big { font-size: 112px; font-weight: 800; color: #2f6f00; letter-spacing: -2px; line-height: 1.05; }
.lbl { font-size: 34px; color: #3d3d3a; margin-top: 14px; line-height: 1.3; }
.note { font-size: 33px; color: #2b2b28; margin-top: 54px; line-height: 1.55; }
.rows .row { display: flex; align-items: center; padding: 26px 0; border-bottom: 2px solid #e6e6e0; }
.rows .name { width: 760px; font-size: 40px; font-weight: 700; padding-left: 26px; border-left: 10px solid #c9c9c2; }
.rows .row.ours .name { border-left-color: #76b900; color: #2f6f00; }
.rows .desc { font-size: 33px; color: #3d3d3a; }
.barrow { display: flex; align-items: center; margin: 30px 0; }
.barrow .who { width: 690px; font-size: 34px; font-weight: 650; }
.barrow .errn { position: absolute; top: 6px; font-size: 36px; font-weight: 800; color: #333; }
.barrow .track { flex: 1; height: 58px; position: relative; }
.barrow .fill { height: 58px; border-radius: 6px; }
.barrow .pct { width: 230px; text-align: right; font-size: 44px; font-weight: 800; }
.barrow .cost { width: 270px; text-align: right; font-size: 32px; color: #4a4a46; }
.hdr { display: flex; font-size: 26px; color: #6b6b66; margin-bottom: -10px; }
.punch { font-size: 36px; font-weight: 700; color: #2f6f00; margin-top: 40px; line-height: 1.45; }
.term { background: #101412; color: #d7e3d0; border-radius: 16px; padding: 24px 36px; font-size: 20px; line-height: 1.34;
        font-family: "SF Mono", Menlo, Monaco, monospace; white-space: pre; }
.term .ok { color: #8fd14f; } .term .bad { color: #ff7b72; } .term .dim { color: #8b978a; }
.two { display: flex; gap: 80px; }
.two > div { flex: 1; }
.card { background: #fff; border: 2px solid #e2e2dc; border-radius: 18px; padding: 40px 44px; }
.card h3 { font-size: 38px; margin-bottom: 26px; }
.cmp { display: flex; justify-content: space-between; align-items: baseline; font-size: 34px; padding: 16px 0; border-bottom: 2px solid #efefe9; }
.cmp span:first-child { flex: 1; padding-right: 24px; }
.cmp span:last-child, .cmp b { white-space: nowrap; }
.cmp b { color: #2f6f00; }
.small { font-size: 28px; color: #4a4a46; margin-top: 26px; line-height: 1.45; }
"""


def page(body: str, num: int, total: int) -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
            f"<div class='bar'></div><div class='wrap'>{body}</div>"
            f"<div class='foot'><span>PhishGuard · Fireworks Deploy · NVIDIA Best-Use track</span>"
            f"<span>{num:02d} / {total:02d}</span></div></body></html>")


def bars() -> str:
    """Errors on the 600 held-out cases: shorter is better, and the axis starts at zero."""
    rows = [("Nemotron 3.5 Super + LoRA (ours)", 13, 97.8, "$233", "#76b900"),
            ("Kimi K3, 2.8T (frontier)", 23, 96.2, "$1,600", "#eb6834"),
            ("DeepSeek V4.1 Flash, 552B", 39, 93.5, "$114", "#1baf7a"),
            ("Nemotron 3.5 Super, untuned", 56, 90.7, "$233", "#9a9a92")]
    out = ["<div class='hdr'><span style='width:690px'></span><span style='flex:1'>errors out of 600 (lower is better)</span>"
           "<span style='width:230px;text-align:right'>accuracy</span><span style='width:270px;text-align:right'>$ / 1M msgs</span></div>"]
    for who, errs, pct, cost, color in rows:
        w = errs / 60 * 100
        out.append(f"<div class='barrow'><div class='who'>{who}</div><div class='track'>"
                   f"<div class='fill' style='width:{w:.1f}%;background:{color}'></div>"
                   f"<span class='errn' style='left:calc({w:.1f}% + 14px)'>{errs}</span></div>"
                   f"<div class='pct'>{pct:.1f}%</div><div class='cost'>{cost}</div></div>")
    return "\n".join(out)


DEMO = [
    ("[SMS]  truth: MALICIOUS", "Here is your discount code RP176781. To stop further messages reply stop…",
     [("ok", "ours   MALICIOUS   recorded   $156 / 1M msgs"), ("ok", "Kimi K3 MALICIOUS  1.33 s     $1,218 / 1M msgs")]),
    ("[EMAIL]  truth: benign", "Subject: RE: LOCATION FOR MAN NIGHT  Either way, it sounds like…",
     [("ok", "ours   benign      recorded   $518 / 1M msgs"), ("ok", "Kimi K3 benign     0.91 s     $2,973 / 1M msgs")]),
    ("[URL]  truth: MALICIOUS", "https://www.citieid.com/kenya",
     [("bad", "ours   benign      recorded   $153 / 1M msgs"), ("bad", "Kimi K3 benign     0.69 s     $1,212 / 1M msgs")]),
    ("[CALL, first 50%]  truth: MALICIOUS", "caller: Hello, this is Karen from the Microsoft Refund Department…",
     [("ok", "ours   MALICIOUS   recorded   $275 / 1M msgs"), ("ok", "Kimi K3 MALICIOUS  1.45 s     $1,836 / 1M msgs")]),
]


def terminal() -> str:
    lines = ["<span class='dim'>$ uv run python scripts/demo.py --models ft-nemotron-super kimi-k3</span>"]
    for head, text, preds in DEMO:
        lines.append("")
        lines.append(html.escape(head))
        lines.append(f"  <span class='dim'>{html.escape(text)}</span>")
        for cls, p in preds:
            mark = "✔" if cls == "ok" else "✘"
            lines.append(f"  <span class='{cls}'>{mark}</span> {html.escape(p)}")
    return "<div class='term'>" + "\n".join(lines) + "</div>"


# (slide body, narration). Numbers: results/summary.md and results/metrics.json.
DECK = [
    ("""<div class='kicker'>Fireworks: Deploy with Microsoft &amp; NVIDIA · Nov 2, 2026</div>
<h1>PhishGuard: a fine-tuned NVIDIA Nemotron<br>beats a frontier model at scam screening</h1>
<div class='rule'></div>
<div class='sub'>Nemotron 3.5 Super + LoRA on Fireworks<br>Email · SMS · URLs · phone-call transcripts · 600 held-out cases</div>
<div class='punch' style='font-size:44px;margin-top:60px'>97.8% vs 96.2% for Kimi K3 on 600 held-out cases (p = 0.041)<br>~7× cheaper · 23× smaller · open model</div>
<div class='note'>Team: John · Kris &nbsp;&nbsp;|&nbsp;&nbsp; """ + REPO + "</div>",
     "This is PhishGuard. We fine-tuned an open NVIDIA Nemotron model on Fireworks, and it beats a frontier model "
     "at screening scams."),
    ("""<h2>Scams hit every channel. Frontier screening costs too much.</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>4</div><div class='lbl'>channels to screen:<br>email, SMS, links, calls</div></div>
 <div class='stat'><div class='big'>$1,600</div><div class='lbl'>per 1M messages with a<br>frontier model (Kimi K3)</div></div>
 <div class='stat'><div class='big'>Now</div><div class='lbl'>a verdict is needed before<br>the click or the call ends</div></div>
</div>
<div class='note'><b>User:</b> fraud and security teams at banks, telcos and enterprises that must screen every message.</div>""",
     "Fraud teams at banks and telcos must screen every email, text, link and call, ideally before the customer clicks. "
     "Frontier models are accurate, but at about sixteen hundred dollars per million messages, they are too expensive "
     "to run on everything."),
    ("""<h2>One evaluation, frozen before any run</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>2,400</div><div class='lbl'>training examples<br>(600 per channel, balanced)</div></div>
 <div class='stat'><div class='big'>600</div><div class='lbl'>held-out test cases<br>(150 per channel)</div></div>
 <div class='stat'><div class='big'>4</div><div class='lbl'>channels, one<br>malicious / benign label</div></div>
</div>
<div class='note'>Plus: 200 emails from an unseen later corpus (2008) · 200 recent synthetic scams · 300 call prefixes<br>
Same hash-checked files and scoring for every system · data from Hugging Face</div>""",
     "So we built one evaluation and froze it: twenty-four hundred training examples, and six hundred held-out test "
     "cases across four channels, plus temporal, out-of-distribution and early-warning sets."),
    ("""<h2>Four systems, the same cases, the same prompt</h2><div class='rule'></div>
<div class='rows'>
 <div class='row'><div class='name'>Kimi K3 (2.8T)</div><div class='desc'>Frontier baseline; metadata + timing cues in the prompt</div></div>
 <div class='row'><div class='name'>DeepSeek V4.1 Flash (552B)</div><div class='desc'>Cost-efficient frontier baseline</div></div>
 <div class='row'><div class='name'>Nemotron 3.5 Super, untuned</div><div class='desc'>The "before": same prompt, same decoding</div></div>
 <div class='row ours'><div class='name'>Nemotron 3.5 Super + LoRA</div><div class='desc'>Ours: trained on the 2,400 examples; prompt unchanged</div></div>
</div>""",
     "Every system answered the same cases with the same prompt: Kimi K3 and DeepSeek as frontier baselines, Nemotron "
     "untuned, and our LoRA fine-tune."),
    ("""<h2>The lightweight training run</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>Rank 16</div><div class='lbl'>LoRA adapter on<br>Nemotron 3.5 Super (120B / 12B active)</div></div>
 <div class='stat'><div class='big'>1 epoch</div><div class='lbl'>75 steps × 32 examples,<br>learning rate 1e-4</div></div>
 <div class='stat'><div class='big'>~11 min</div><div class='lbl'>training time</div></div>
</div>
<div class='note'><b>Training charge for the evaluated epoch: ~$2.20</b> &nbsp;(944k tokens × $2.32 / 1M) · whole project ≈ $8 of $50<br>
Fireworks Serverless Training API: no dataset upload, no deployment; base and tuned adapter sampled in-session</div>""",
     "Training was light: a rank sixteen LoRA adapter, one epoch, about eleven minutes and two dollars, on Fireworks "
     "Serverless Training, with no deployment needed."),
    ("<h2>Held-out accuracy and cost</h2><div class='rule'></div>" + bars() + """
<div class='punch'>+10 net correct vs Kimi K3 (15 vs 5 discordant), McNemar p = 0.041 &nbsp;·&nbsp; ΔF1 +1.7 [+0.2, +3.3]<br>
43% fewer errors than Kimi K3 &nbsp;·&nbsp; +43 correct vs the same model untuned &nbsp;·&nbsp; about 7× cheaper</div>""",
     "The result: ninety-seven point eight percent accuracy, against ninety-six point two for Kimi K3. The gain is "
     "statistically significant, with a McNemar p-value of point zero four. Fine-tuning added forty-three correct "
     "answers over the untuned model, at about one seventh of Kimi's cost."),
    ("<h2>Demo: held-out items, ours vs. the frontier</h2><div class='rule' style='margin-bottom:30px'></div>" + terminal() + """
<div class='small'>Ours: verdicts recorded in the evaluation run (the adapter was sampled in its training session) · Kimi K3: live API calls</div>""",
     "Here are held-out examples: our model's recorded verdicts next to live Kimi K3 calls, with the cost per million "
     "messages on each line. Both models miss the tricky URL."),
    ("""<h2>Timing: early warning, and the honest tradeoff</h2><div class='rule'></div>
<div class='two'>
 <div class='card'><h3>After hearing 25% of a call</h3>
  <div class='cmp'><span>Ours: scams caught</span><b>96%</b></div>
  <div class='cmp'><span>Ours: false alarms</span><b>2%</b></div>
  <div class='cmp'><span>Kimi K3: scams caught</span><span>98%</span></div>
  <div class='cmp'><span>Kimi K3: false alarms</span><span>14%</span></div>
  <div class='small'>From 50% of the call on, ours: 100% caught, 0% false alarms</div></div>
 <div class='card'><h3>Drift: where the frontier wins</h3>
  <div class='cmp' style='font-size:26px;color:#6b6b66'><span></span><span>ours vs Kimi K3</span></div>
  <div class='cmp'><span>Unseen 2008 email corpus, F1</span><span>83.8 vs <b>91.5</b></span></div>
  <div class='cmp'><span>Recent synthetic scams, accuracy</span><span>86.0 vs <b>94.0</b></span></div>
  <div class='small'>Next: refresh the adapter on recent data and route out-of-distribution traffic to a frontier fallback.</div></div>
</div>""",
     "Timing matters too. After hearing only a quarter of a call, our model flags ninety-six percent of scams with two "
     "percent false alarms; Kimi raises fourteen percent. The honest tradeoff: on an unseen 2008 email corpus and on "
     "recent synthetic scams, the frontier model generalizes better, so the adapter needs refreshing with new data."),
    ("""<h2>Why adopt PhishGuard</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>97.8%</div><div class='lbl'>beats the frontier on<br>the target task (p = 0.041)</div></div>
 <div class='stat'><div class='big'>~7×</div><div class='lbl'>cheaper per message<br>than Kimi K3</div></div>
 <div class='stat'><div class='big'>23×</div><div class='lbl'>smaller model<br>(120B vs 2.8T)</div></div>
</div>
<div class='note'>An open NVIDIA Nemotron model you can self-host, so customer messages stay on your infrastructure.<br>
Reproduce everything: <b>""" + REPO + "</b> &nbsp;(make data · make report)</div>",
     "PhishGuard: frontier-beating accuracy on the target task, seven times cheaper, twenty-three times smaller, and an "
     "open NVIDIA model you can run on your own infrastructure. The code and the evidence are in our repo."),
]


# Final narration, trimmed so the whole video stays well under two minutes (one line per slide above).
NARRATION = [
    "This is PhishGuard: an open NVIDIA Nemotron model, fine-tuned on Fireworks, that beats a frontier model at screening scams.",
    "Fraud teams must screen every email, text, link and call before customers act. Frontier models are accurate, but at "
    "about sixteen hundred dollars per million messages, they are too expensive to run on everything.",
    "We froze one evaluation: twenty-four hundred training examples, and six hundred held-out cases across four channels, "
    "plus drift and early-warning sets.",
    "Every system answered the same cases with the same prompt: Kimi K3 and DeepSeek as frontier baselines, Nemotron "
    "untuned, and our fine-tune.",
    "Training was light: a rank sixteen LoRA, one epoch, about eleven minutes and two dollars on Fireworks Serverless Training.",
    "The result: ninety-seven point eight percent accuracy, versus ninety-six point two for Kimi K3. That is forty-three "
    "percent fewer errors, significant at p equals point zero four, and about seven times cheaper.",
    "On held-out examples, our recorded verdicts match live Kimi K3 calls at a fraction of the cost. Both miss the tricky URL.",
    "After hearing a quarter of a call, we flag ninety-six percent of scams with two percent false alarms; Kimi raises "
    "fourteen percent. The tradeoff: on a later, unseen email corpus and on new synthetic scams, the frontier generalizes "
    "better, so we would refresh the adapter with new data.",
    "PhishGuard: frontier-beating accuracy, seven times cheaper, twenty-three times smaller, and an open NVIDIA model you "
    "can run yourself. The evidence is in our repo.",
]
DECK = [(body, text) for (body, _), text in zip(DECK, NARRATION)]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_slides() -> list[Path]:
    SLIDES.mkdir(parents=True, exist_ok=True)
    (BUILD / "html").mkdir(parents=True, exist_ok=True)
    pngs = []
    for i, (body, _) in enumerate(DECK, 1):
        src = BUILD / "html" / f"slide_{i:02d}.html"
        src.write_text(page(body, i, len(DECK)))
        png = SLIDES / f"slide_{i:02d}.png"
        run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
             "--window-size=1920,1080", f"--screenshot={png}", src.as_uri()])
        pngs.append(png)
    images = [Image.open(p).convert("RGB") for p in pngs]
    images[0].save(DOCS / "slides.pdf", save_all=True, append_images=images[1:], resolution=144)
    print(f"slides: {len(pngs)} PNGs in {SLIDES.relative_to(ROOT)}, PDF at docs/slides.pdf")
    return pngs


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def build_video(pngs: list[Path]) -> None:
    (BUILD / "audio").mkdir(parents=True, exist_ok=True)
    segs, script, total = [], ["# Video script (narration, ~2 min)", "",
                               "Generated by `scripts/make_media.py`; voice: macOS `say -v Samantha`.", ""], 0.0
    for i, ((_, text), png) in enumerate(zip(DECK, pngs), 1):
        aiff = BUILD / "audio" / f"slide_{i:02d}.aiff"
        run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text])
        seg = BUILD / f"seg_{i:02d}.mp4"
        run(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i", str(png), "-i", str(aiff),
             "-af", "apad=pad_dur=0.45", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", str(seg)])
        d = duration(seg)
        script += [f"**Slide {i} ({total:.0f}–{total + d:.0f} s).** {text}", ""]
        total += d
        segs.append(seg)
    concat = BUILD / "concat.txt"
    concat.write_text("".join(f"file '{s}'\n" for s in segs))
    out = DOCS / "video" / "phishguard_2min.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(out)])
    (DOCS / "VIDEO_SCRIPT.md").write_text("\n".join(script) + f"\nTotal: {duration(out):.1f} s\n")
    print(f"video: {out.relative_to(ROOT)} ({duration(out):.1f} s, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", action="store_true", help="slides only")
    args = ap.parse_args()
    pngs = build_slides()
    if not args.slides:
        build_video(pngs)

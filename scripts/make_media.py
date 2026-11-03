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
import re
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
.sec { display: inline-block; font-size: 24px; font-weight: 750; letter-spacing: 2px; color: #fff; background: #2f6f00;
       padding: 6px 16px; border-radius: 6px; margin-bottom: 22px; text-transform: uppercase; }
.code { background: #0f1412; color: #dfe7da; border-radius: 16px; padding: 30px 38px; font-size: 25px; line-height: 1.45;
        font-family: "SF Mono", Menlo, Monaco, monospace; white-space: pre; }
.code .k { color: #c792ea; } .code .s { color: #c3e88d; } .code .c { color: #7f8c7a; font-style: italic; } .code .n { color: #f78c6c; }
.src { font-size: 24px; color: #6b6b66; margin-top: 16px; }
.ba { width: 100%; border-collapse: collapse; font-size: 34px; }
.ba td { padding: 15px 10px; border-bottom: 2px solid #e6e6e0; }
.ba td:nth-child(2), .ba td:nth-child(3) { text-align: right; white-space: nowrap; }
.ba td:nth-child(3) { color: #2f6f00; font-weight: 800; }
.ba tr.head td { font-size: 26px; color: #6b6b66; font-weight: 600; }
.ba tr.reg td:nth-child(3) { color: #b3261e; }
.hero { font-size: 104px; font-weight: 850; letter-spacing: -2px; color: #111; margin: 4px 0 26px; }
.hero b { color: #2f6f00; }
.stack .row { display: flex; padding: 20px 0; border-bottom: 2px solid #e6e6e0; align-items: baseline; }
.stack .name { width: 600px; font-size: 36px; font-weight: 750; color: #2f6f00; }
.stack .desc { flex: 1; font-size: 31px; color: #2b2b28; line-height: 1.35; }
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


KW = re.compile(r"\b(def|for|in|return|import|from|lambda|True|False|None)\b")


def highlight(code: str) -> str:
    out = []
    for line in code.strip("\n").split("\n"):
        body, hash_, comment = line.partition("#")
        body = html.escape(body, quote=False)
        body = re.sub(r"(\"[^\"]*\"|'[^']*')", r"<span class='s'>\1</span>", body)
        body = KW.sub(r"<span class='k'>\1</span>", body)
        body = re.sub(r"(?<![\w.'\"-])(\d+(?:\.\d+)?(?:e-?\d+)?)\b", r"<span class='n'>\1</span>", body)
        out.append(body + (f"<span class='c'>#{html.escape(comment, quote=False)}</span>" if hash_ else ""))
    return "<div class='code'>" + "\n".join(out) + "</div>"


def terminal(upto: int = len(DEMO)) -> str:
    lines = ["<span class='dim'>$ uv run python scripts/demo.py --models ft-nemotron-super kimi-k3</span>"]
    for head, text, preds in DEMO[:upto]:
        lines.append("")
        lines.append(html.escape(head))
        lines.append(f"  <span class='dim'>{html.escape(text)}</span>")
        for cls, p in preds:
            mark = "✔" if cls == "ok" else "✘"
            lines.append(f"  <span class='{cls}'>{mark}</span> {html.escape(p)}")
    return "<div class='term'>" + "\n".join(lines) + "</div>"


TRAIN_CODE = """
svc = FiretitanServiceClient(api_key=KEY, base_url="https://api.fireworks.ai/training/v1/serverless")
tc = svc.create_lora_training_client(base_model=NEMOTRON_SUPER_120B, rank=16)

def to_datum(tok, row):                      # train only on the answer tokens
    prompt = render(tok, row["messages"][:-1], add_generation_prompt=True)
    full = render(tok, row["messages"], add_generation_prompt=False)
    w = [0.0] * len(prompt) + [1.0] * (len(full) - len(prompt))
    return Datum(model_input=ModelInput.from_ints(full[:-1]),
                 loss_fn_inputs={"target_tokens": full[1:], "weights": w[1:]})

for batch in batches(datums, size=32):       # 75 steps = 1 epoch, ~11 min
    tc.forward_backward(batch, "cross_entropy")
    tc.optim_step(AdamParams(learning_rate=1e-4)).result()
ckpt = tc.save_weights_for_sampler("phish-ep1").result().path   # sampled in-session: no deploy
"""

EVAL_CODE = """
meta = {**link_summary(text),                # domains, shorteners, IP hosts, punycode
        **content_cues(text),                # urgency words, deadlines, OTP / gift-card asks
        **time_features(sent_at)}            # send hour, off-hours
user = f"Channel: {channel}\\nMetadata:\\n{format_meta(meta)}\\nContent:\\n{text}"
# every model returns {"label": "malicious" | "benign"} (JSON schema, temperature 0)

# same 600 frozen cases for every system, then a paired McNemar test
b = sum(ours_ok & ~kimi_ok)                  # only ours right: 15
c = sum(kimi_ok & ~ours_ok)                  # only Kimi K3 right: 5
p = binomtest(b, b + c, 0.5).pvalue          # p = 0.041
"""


def sec(label: str) -> str:
    return f"<div class='sec'>{label}</div>"


# (slide body or list of animation frames, narration). Numbers: results/summary.md, results/metrics.json.
DECK = [
    (sec("PhishGuard") + """<h1>A fine-tuned NVIDIA Nemotron<br>that catches scams on every channel</h1>
<div class='rule'></div>
<div class='hero' style='font-size:84px'>Fine-tuning: 90.7% <b>&rarr; 97.8%</b></div>
<div class='sub'>Beats the frontier model Kimi K3 (96.2%) on 600 held-out cases · about 7× cheaper</div>
<div class='note'>Team: John Shi · Kris Chai &nbsp;&nbsp;|&nbsp;&nbsp; """ + REPO + "</div>",
     "This is PhishGuard: an NVIDIA Nemotron model, fine-tuned on Fireworks, that catches scams across every channel."),
    (sec("1 · The problem") + """<h2>Scams hit every channel. Screening them all is too expensive.</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>4</div><div class='lbl'>channels to screen:<br>email, SMS, links, calls</div></div>
 <div class='stat'><div class='big'>$1,600</div><div class='lbl'>per 1M messages with a<br>frontier model (Kimi K3)</div></div>
 <div class='stat'><div class='big'>Now</div><div class='lbl'>a verdict is needed before<br>the click or the call ends</div></div>
</div>
<div class='note'><b>Who it is for:</b> fraud and security teams at banks, telcos and enterprises that must screen every message, fast, without sending customer data to a third party.</div>""",
     "The problem: fraud teams must screen every email, text, link and phone call before customers act. Frontier models "
     "are accurate, but at sixteen hundred dollars per million messages, they are too expensive to run on everything."),
    (sec("2 · Tech stack") + """<h2>What the stack made possible</h2><div class='rule' style='margin-bottom:20px'></div>
<div class='stack'>
 <div class='row'><div class='name'>Fireworks Serverless Training</div><div class='desc'>LoRA fine-tune of a 120B model in ~11 min for ~$2.20: no GPUs, no dataset upload, no deployment</div></div>
 <div class='row'><div class='name'>Fireworks serverless inference</div><div class='desc'>Frontier baselines (Kimi K3 2.8T, DeepSeek V4.1 Flash) through one API, JSON-schema output</div></div>
 <div class='row'><div class='name'>NVIDIA Nemotron 3.5 Super</div><div class='desc'>Open 120B MoE (12B active): strong base, self-hostable for private screening</div></div>
 <div class='row'><div class='name'>Hugging Face + Python</div><div class='desc'>5 public datasets across 4 channels; async eval harness, bootstrap CIs, McNemar tests</div></div>
</div>
<div class='punch' style='font-size:32px;margin-top:30px'>Impossible without it: tuning a 120B model with no GPUs in minutes, then scoring it against a 2.8T frontier model through the same API.</div>""",
     "Our stack: Fireworks Serverless Training fine-tuned a hundred-and-twenty-billion-parameter Nemotron with LoRA in "
     "eleven minutes for about two dollars, with no GPUs, uploads or deployment. Fireworks inference gave us frontier "
     "baselines through one API, and Hugging Face supplied the data."),
    (sec("2 · Tech stack: the evaluation") + """<h2>One evaluation, frozen before any run</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>2,400</div><div class='lbl'>training examples<br>(600 per channel)</div></div>
 <div class='stat'><div class='big'>600</div><div class='lbl'>held-out test cases<br>(150 per channel)</div></div>
 <div class='stat'><div class='big'>4</div><div class='lbl'>systems, same prompt,<br>same scoring</div></div>
</div>
<div class='note'>Kimi K3 (frontier) · DeepSeek V4.1 Flash · Nemotron 3.5 Super untuned (<b>before</b>) · Nemotron 3.5 Super + LoRA (<b>after</b>)<br>
Plus 200 later-corpus emails, 200 recent synthetic scams and 300 call prefixes for drift and early warning</div>""",
     "We froze one evaluation: twenty-four hundred training examples and six hundred held-out cases, scored "
     "identically for every system."),
    ([sec("3 · Live demo") + "<h2>Live demo: held-out messages</h2><div class='rule' style='margin-bottom:30px'></div>" + terminal(k)
      + "<div class='small'>Ours: verdicts recorded in the evaluation run (adapter sampled in its training session) · Kimi K3: live API calls</div>"
      for k in range(1, len(DEMO) + 1)],
     "Here it is in action on held-out messages: our verdicts next to live Kimi K3 calls, with the cost per million "
     "messages on each line. Both miss one tricky URL."),
    (sec("3 · Code: fine-tuning") + "<h2>The training loop</h2><div class='rule' style='margin-bottom:26px'></div>" + highlight(TRAIN_CODE)
     + "<div class='src'>Excerpt, simplified from scripts/train_serverless.py</div>",
     "And the code. Training is a short loop: create a LoRA client, put the loss only on the answer tokens, then step "
     "and save a checkpoint we can sample directly."),
    (sec("3 · Code: features and scoring") + "<h2>Metadata in, label out, paired scoring</h2><div class='rule' style='margin-bottom:26px'></div>"
     + highlight(EVAL_CODE) + "<div class='src'>Excerpts from src/phish/features.py, prompt.py and report.py</div>",
     "Each message gets cheap metadata and timing cues, like urgency words and suspicious links, and every model is "
     "scored on the same cases with a paired test."),
    (sec("Results: before vs after fine-tuning") + """<h2>Same model, same prompt: before vs after fine-tuning</h2><div class='rule' style='margin-bottom:18px'></div>
<div class='hero'>90.7% <b>&rarr; 97.8%</b> <span style='font-size:44px;color:#4a4a46;font-weight:600'>accuracy on 600 held-out cases</span></div>
<table class='ba'>
 <tr class='head'><td>Nemotron 3.5 Super</td><td>before</td><td>after</td></tr>
 <tr><td>Errors out of 600</td><td>56</td><td>13 (−77%)</td></tr>
 <tr><td>False-positive rate</td><td>15.0%</td><td>1.0%</td></tr>
 <tr><td>Accuracy by channel: email · SMS · URL · calls</td><td>94 · 94 · 92 · 83</td><td>97 · 97 · 97 · 100</td></tr>
 <tr><td>Call false alarms after hearing 25% of the call</td><td>34%</td><td>2%</td></tr>
 <tr class='reg'><td>Later, unseen email corpus (regression)</td><td>87.0%</td><td>85.5%</td></tr>
</table>""",
     "Now the key result. Before fine-tuning, Nemotron scored ninety point seven percent. After fine-tuning, ninety-seven "
     "point eight. Errors fell from fifty-six to thirteen, and false alarms on phone calls dropped from thirty-four "
     "percent to two."),
    (sec("Results: vs the frontier") + "<h2>After fine-tuning, it beats the frontier</h2><div class='rule'></div>" + bars() + """
<div class='punch'>13 vs 23 errors: McNemar p = 0.041 · ΔF1 +1.7 [+0.2, +3.3] · about 7× cheaper than Kimi K3<br>
<span style='color:#b3261e'>Tradeoff: Kimi K3 generalizes better to newer, unseen data (2008 corpus F1 91.5 vs 83.8)</span></div>""",
     "That also beats the frontier: thirteen errors versus twenty-three for Kimi K3, significant at p equals point zero "
     "four, at one seventh of the cost. The tradeoff: Kimi generalizes better to newer, unseen data."),
    (sec("Why adopt") + """<h2>PhishGuard</h2><div class='rule'></div>
<div class='stats'>
 <div class='stat'><div class='big'>+7.1</div><div class='lbl'>accuracy points from<br>fine-tuning (90.7 &rarr; 97.8)</div></div>
 <div class='stat'><div class='big'>~7×</div><div class='lbl'>cheaper per message<br>than Kimi K3</div></div>
 <div class='stat'><div class='big'>23×</div><div class='lbl'>smaller model<br>(120B vs 2.8T)</div></div>
</div>
<div class='note'>An open NVIDIA Nemotron model you can self-host, so customer messages stay on your infrastructure.<br>
Reproduce everything: <b>""" + REPO + "</b></div>",
     "PhishGuard: frontier-beating accuracy from an open NVIDIA model you can run yourself. The code and evidence are in "
     "our repo."),
]

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def screenshot(html_text: str, name: str, png: Path) -> Path:
    src = BUILD / "html" / f"{name}.html"
    src.write_text(html_text)
    run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
         "--window-size=1920,1080", f"--screenshot={png}", src.as_uri()])
    return png


def build_slides() -> list[list[Path]]:
    """One PNG per slide in docs/slides (final frame); animation frames for the video go to the build dir."""
    for old in SLIDES.glob("slide_*.png"):
        old.unlink()
    SLIDES.mkdir(parents=True, exist_ok=True)
    (BUILD / "html").mkdir(parents=True, exist_ok=True)
    (BUILD / "frames").mkdir(parents=True, exist_ok=True)
    frames_per_slide = []
    for i, (body, _) in enumerate(DECK, 1):
        bodies = body if isinstance(body, list) else [body]
        frames = [screenshot(page(b, i, len(DECK)), f"slide_{i:02d}_{k}", BUILD / "frames" / f"slide_{i:02d}_{k}.png")
                  for k, b in enumerate(bodies, 1)]
        final = SLIDES / f"slide_{i:02d}.png"
        final.write_bytes(frames[-1].read_bytes())
        frames_per_slide.append(frames)
    images = [Image.open(SLIDES / f"slide_{i:02d}.png").convert("RGB") for i in range(1, len(DECK) + 1)]
    images[0].save(DOCS / "slides.pdf", save_all=True, append_images=images[1:], resolution=144)
    print(f"slides: {len(DECK)} PNGs in {SLIDES.relative_to(ROOT)}, PDF at docs/slides.pdf")
    return frames_per_slide


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def build_video(pngs: list[list[Path]]) -> None:
    (BUILD / "audio").mkdir(parents=True, exist_ok=True)
    segs, script, total = [], ["# Video script (narration, ~2 min)", "",
                               "Generated by `scripts/make_media.py`; voice: macOS `say -v Samantha`.", ""], 0.0
    for i, ((_, text), frames) in enumerate(zip(DECK, pngs), 1):
        aiff = BUILD / "audio" / f"slide_{i:02d}.aiff"
        run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text])
        seg = BUILD / f"seg_{i:02d}.mp4"
        # Animated slides split the narration evenly across their frames; static slides hold one image.
        step = (duration(aiff) + 0.45) / len(frames)
        listing = BUILD / f"frames_{i:02d}.txt"
        listing.write_text("".join(f"file '{f}'\nduration {step:.3f}\n" for f in frames) + f"file '{frames[-1]}'\n")
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(aiff),
             "-af", "apad=pad_dur=0.45", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "30",
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

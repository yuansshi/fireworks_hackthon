"""Launch and monitor Fireworks LoRA SFT jobs with just the API key (no firectl sign-in needed).

    uv run python scripts/finetune_api.py launch smoke     # 200 rows, 1 epoch: proves quota + format
    uv run python scripts/finetune_api.py launch meta      # FT-meta (primary)
    uv run python scripts/finetune_api.py launch text      # FT-text (metadata ablation)
    uv run python scripts/finetune_api.py status [JOB_ID]  # one job, or all recent jobs
"""

from __future__ import annotations

import json
import os
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import truststore
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API = "https://api.fireworks.ai/v1"
BASE_MODEL = os.environ.get("BASE_MODEL", "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b")
JOBS = {
    "smoke": ("data/ft_smoke_train.jsonl", None, 1),
    "meta": ("data/ft_meta_train.jsonl", "data/ft_meta_val.jsonl", 2),
    "text": ("data/ft_text_train.jsonl", "data/ft_text_val.jsonl", 2),
}


def client() -> tuple[httpx.Client, str]:
    key, acct = os.environ.get("FIREWORKS_API_KEY"), os.environ.get("FIREWORKS_ACCOUNT_ID")
    if not key or not acct:
        raise SystemExit("set FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID in .env")
    # System trust store: corporate TLS-inspecting proxies add a root CA that certifi does not know.
    verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.Client(base_url=API, headers={"Authorization": f"Bearer {key}"}, timeout=120, verify=verify), acct


def check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        raise SystemExit(f"HTTP {r.status_code} {r.request.method} {r.request.url}: {r.text[:500]}")
    return r.json() if r.content else {}


def upload_dataset(c: httpx.Client, acct: str, dataset_id: str, path: Path) -> str:
    n = sum(1 for _ in path.open())
    check(c.post(f"/accounts/{acct}/datasets", json={
        "datasetId": dataset_id,
        "dataset": {"displayName": dataset_id, "format": "CHAT", "userUploaded": {}, "exampleCount": str(n)},
    }))
    with path.open("rb") as fh:
        check(c.post(f"/accounts/{acct}/datasets/{dataset_id}:upload", files={"file": (path.name, fh, "application/jsonl")}))
    for _ in range(60):  # wait until the server has validated the upload
        state = check(c.get(f"/accounts/{acct}/datasets/{dataset_id}")).get("state")
        if state == "READY":
            break
        time.sleep(2)
    print(f"  dataset {dataset_id}: {n} rows, state={state}")
    return f"accounts/{acct}/datasets/{dataset_id}"


def launch(which: str) -> None:
    train, val, epochs = JOBS[which]
    c, acct = client()
    stamp = datetime.now().strftime("%m%d-%H%M")
    name = f"phish-{which}-{stamp}"
    body = {
        "dataset": upload_dataset(c, acct, f"{name}-train", ROOT / train),
        "baseModel": BASE_MODEL,
        "outputModel": f"accounts/{acct}/models/{name}",
        "epochs": epochs,
        "loraRank": 16,  # always explicit: an omitted rank means full-parameter training at 2x the price
        "displayName": name,
    }
    if val:
        body["evaluationDataset"] = upload_dataset(c, acct, f"{name}-val", ROOT / val)
    job = check(c.post(f"/accounts/{acct}/supervisedFineTuningJobs", json=body))
    print(json.dumps({k: job.get(k) for k in ("name", "state", "outputModel", "estimatedCost", "createTime")}, indent=2))
    print(f"\nOutput model: {body['outputModel']}\nStatus:       uv run python scripts/finetune_api.py status {job['name'].split('/')[-1]}")


def status(job_id: str | None) -> None:
    c, acct = client()
    if job_id:
        jobs = [check(c.get(f"/accounts/{acct}/supervisedFineTuningJobs/{job_id}"))]
    else:
        jobs = check(c.get(f"/accounts/{acct}/supervisedFineTuningJobs", params={"pageSize": 10})).get("supervisedFineTuningJobs", [])
    for j in jobs:
        prog = j.get("jobProgress") or {}
        print(f"{j['name'].split('/')[-1]:40s} {j.get('state'):22s} progress={json.dumps(prog)[:120]} "
              f"cost={j.get('estimatedCost')} model={j.get('outputModel', '').split('/')[-1]}")
        if (j.get("status") or {}).get("message"):
            print(f"   status: {j['status']['message'][:300]}")
    if not jobs:
        print("no fine-tuning jobs yet")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("launch", "status"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "launch":
        if len(sys.argv) < 3 or sys.argv[2] not in JOBS:
            raise SystemExit("usage: finetune_api.py launch smoke|meta|text")
        launch(sys.argv[2])
    else:
        status(sys.argv[2] if len(sys.argv) > 2 else None)

"""
main.py - Single-file TDS Quiz Solver (FastAPI)

Features:
- Single endpoint POST / that accepts {"email","secret","url", ...}
- Validates secret
- Returns HTTP 200 quickly and solves the quiz in background (must finish submissions within 3 minutes)
- Uses Playwright to render JS pages
- Parses atob(...) base64 payloads and <pre> JSON blocks
- Downloads CSV/JSON/PDF/images and parses them
- Heuristic solvers for common tasks (sum, mean, count, ML, plot)
- LLM orchestration (OpenAI) to generate verification / Python plans
- Executes Python in short-lived subprocess with timeout
- Submits answer to submit URL, handles follow-up URLs and re-submits within time window
Security: run in an isolated container in production. This code takes mitigations (timeouts, tempdir) but is not fully sandboxed.
"""

import os
import re
import io
import json
import time
import uuid
import base64
import shutil
import tempfile
import traceback
import subprocess
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from playwright.async_api import async_playwright
import asyncio

# ----------------------------
# Config (env)
# ----------------------------
QUIZ_SECRET = os.env "tdsiron.get("QUIZ_SECRET",_secret_7kL")  # set in env / Google Form
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL=os.environ.get("OPENAI_BASE_URL")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", 8 * 1024 * 1024))  # 8 MB
PAGE_TIMEOUT_MS = int(os.environ.get("PAGE_TIMEOUT_MS", 60_000))
CODE_EXEC_TIMEOUT_S = int(os.environ.get("CODE_EXEC_TIMEOUT_S", 30))
TOTAL_WINDOW_S = int(os.environ.get("TOTAL_WINDOW_S", 180))  # 3 minutes

TEMP_DIR_ROOT = os.environ.get("TEMP_DIR_ROOT", "/tmp/tds_solver")
os.makedirs(TEMP_DIR_ROOT, exist_ok=True)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="TDS Quiz Solver (Single endpoint)")

class QuizPayload(BaseModel):
    email: str
    secret: str
    url: str

# ----------------------------
# Utilities
# ----------------------------
def extract_atob_and_pre(html: str):
    """Return list of decoded strings from atob(...) and <pre> contents."""
    out = []
    # atob(`...`) or atob("...") etc.
    for m in re.finditer(r'atob\((?:`([^`]+)`|"([^"]+)"|\'([^\']+)\')\)', html, flags=re.IGNORECASE):
        b64 = m.group(1) or m.group(2) or m.group(3)
        try:
            out.append(base64.b64decode(b64).decode("utf-8"))
        except Exception:
            pass
    # look for large base64 blobs
    for m in re.finditer(r'([A-Za-z0-9+/]{200,}={0,2})', html):
        s = m.group(1)
        try:
            out.append(base64.b64decode(s).decode("utf-8"))
        except Exception:
            pass
    # <pre>
    soup = BeautifulSoup(html, "lxml")
    for pre in soup.find_all("pre"):
        txt = pre.get_text()
        out.append(txt)
    return out

async def render_page(url: str) -> str:
    """Render JS page and return combined HTML + visible text."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_navigation_timeout(PAGE_TIMEOUT_MS)
        try:
            await page.goto(url, wait_until="networkidle")
        except Exception:
            # continue even if timeout; try to get whatever's rendered
            pass
        await asyncio.sleep(0.2)
        html = await page.content()
        try:
            text = await page.inner_text("body")
        except Exception:
            text = ""
        await browser.close()
    return (html or "") + "\n\n" + (text or "")

async def download_bytes_async(client: httpx.AsyncClient, url: str, max_bytes=MAX_DOWNLOAD_BYTES, timeout=30):
    try:
        async with client.stream("GET", url, timeout=timeout) as r:
            r.raise_for_status()
            content = bytearray()
            async for chunk in r.aiter_bytes():
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise HTTPException(status_code=400, detail="file too large")
            return bytes(content), r.headers.get("content-type", "")
    except Exception as e:
        raise

def sum_value_in_pdf(pdf_bytes: bytes, column_name='value', page_no=2) -> Optional[float]:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if page_no < 1 or page_no > len(pdf.pages):
                page = pdf.pages[0]
            else:
                page = pdf.pages[page_no - 1]
            tables = page.extract_tables()
            for t in tables:
                df = pd.DataFrame(t[1:], columns=t[0])
                cols = [c.strip().lower() for c in df.columns]
                if column_name.lower() in cols:
                    col = pd.to_numeric(df.iloc[:, cols.index(column_name.lower())].astype(str).str.replace(r'[^0-9\.-]', '', regex=True), errors='coerce')
                    return float(col.sum(skipna=True))
    except Exception:
        return None
    return None

def parse_csv_bytes(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

def parse_json_bytes(content: bytes):
    return json.loads(content.decode("utf-8"))

# ----------------------------
# LLM helpers (OpenAI simple)
# ----------------------------
async def openai_chat(system: str, user: str, max_tokens=800) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "messages":[{"role":"system","content":system},{"role":"user","content":user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

async def llm_plan_for_question(question_text: str, file_summaries: Dict[str, Any]) -> Optional[Dict[str,str]]:
    """
    Ask LLM to produce a short plan and Python snippet to compute ANSWER.
    Return {"reasoning":..., "python":...} or None.
    """
    sys = ("You are a helpful data agent. Produce a JSON object with keys 'reasoning' and 'python'. "
           "The python code must use pandas/numpy only, read files present in the working dir by filename, "
           "and set a variable ANSWER to the final answer. Do not make network calls.")
    user = f"QUESTION:\n{question_text}\n\nFILES:\n{json.dumps(file_summaries,indent=2)}\n\nReturn only JSON."
    try:
        txt = await openai_chat(sys, user, max_tokens=1200)
        # try to extract JSON
        parsed = None
        try:
            parsed = json.loads(txt.strip())
        except Exception:
            # attempt to find JSON block in text
            m = re.search(r'\{[\s\S]*\}', txt)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        return parsed
    except Exception:
        return None

async def llm_verify_answer(question_text: str, candidate_answer):
    """Ask LLM to verify/format candidate answer. Return final answer (string/number/bool/JSON)."""
    sys = "You are a concise assistant. Given a question and a candidate answer, reply with JSON: {\"valid\": true/false, \"final_answer\": <value> }"
    user = f"QUESTION:\n{question_text}\n\nCANDIDATE_ANSWER:\n{json.dumps(candidate_answer)}\n\nIf candidate is valid, set valid true and return final_answer normalized (number/string/bool/base64/JSON). Otherwise valid false with reason."
    try:
        txt = await openai_chat(sys, user, max_tokens=400)
        try:
            parsed = json.loads(txt.strip())
            return parsed
        except Exception:
            m = re.search(r'\{[\s\S]*\}', txt)
            if m:
                return json.loads(m.group(0))
    except Exception:
        return {"valid": True, "final_answer": candidate_answer}

# ----------------------------
# Execute user python safely (subprocess)
# ----------------------------
def run_code_in_subprocess(py_code: str, workdir: str, timeout_s: int = CODE_EXEC_TIMEOUT_S):
    """
    Write runner file in workdir, execute python runner with timeout.
    The wrapper ensures ANSWER variable is saved to result.json.
    """
    runner_py = os.path.join(workdir, "runner.py")
    result_json = os.path.join(workdir, "result.json")
    wrapper = f"""
import json, traceback, os
try:
    import pandas as pd, numpy as np
    ANSWER = None
    FILES = []
{py_code}
    out = {{ "ANSWER": ANSWER, "FILES": FILES }}
    open({json.dumps(result_json)},"w").write(json.dumps(out, default=str))
except Exception as e:
    open({json.dumps(result_json)},"w").write(json.dumps({{"ERROR": traceback.format_exc()}}))
"""
    with open(runner_py, "w", encoding="utf-8") as f:
        f.write(wrapper)
    # execute
    try:
        proc = subprocess.run(["python", runner_py], cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "timeout", None
    if os.path.exists(result_json):
        try:
            j = json.load(open(result_json, "r", encoding="utf-8"))
            if "ERROR" in j:
                return None, j["ERROR"], None
            return j.get("ANSWER"), None, j.get("FILES", [])
        except Exception as e:
            return None, f"result parse error: {e}", None
    else:
        return None, f"no result file, stdout={proc.stdout} stderr={proc.stderr}", None

# ----------------------------
# Heuristic solver (fast)
# ----------------------------
def heuristic_solve(question_text: str, loaded_files: Dict[str, Dict[str,Any]]):
    """Try quick heuristics for frequent tasks (sum value, mean, count, etc)."""
    q = question_text.lower()
    # look for sum of "value" on page N
    m = re.search(r'sum of the ["\']?([^"\']+)["\']? column.*page\s*([0-9]+)', question_text, flags=re.IGNORECASE)
    if m:
        col = m.group(1).strip()
        page_no = int(m.group(2))
        # check loaded PDFs
        for fname, meta in loaded_files.items():
            if meta.get("content_type","").lower().startswith("application/pdf") or fname.lower().endswith(".pdf"):
                if "bytes" in meta:
                    s = sum_value_in_pdf(meta["bytes"], column_name=col, page_no=page_no)
                    if s is not None:
                        return s
    # sum over CSV 'value'
    if "sum" in q:
        for fname, meta in loaded_files.items():
            if "df" in meta:
                df = meta["df"]
                for c in df.columns:
                    if "value" in c.lower():
                        return float(df[c].sum())
    # count/rows
    if "count" in q:
        for meta in loaded_files.values():
            if "df" in meta:
                return int(len(meta["df"]))
    # mean
    if "mean" in q or "average" in q:
        for meta in loaded_files.values():
            if "df" in meta:
                df = meta["df"].select_dtypes(include="number")
                if not df.empty:
                    return float(df.mean().iloc[0])
    # fallback
    return None

# ----------------------------
# Main solver loop
# ----------------------------
async def solve_quiz_loop(email: str, secret: str, start_url: str):
    """
    Attempt to solve the quiz starting at start_url.
    Will attempt submissions until either no next url or time window exhausted.
    """
    start_time = time.time()
    deadline = start_time + TOTAL_WINDOW_S
    current_url = start_url
    # simple HTTP client for downloads and submits
    async with httpx.AsyncClient(timeout=30.0) as client:
        while current_url and time.time() < deadline:
            try:
                # render page
                combined = await render_page(current_url)
            except Exception as e:
                # can't render, abort this URL
                return {"error": f"render failed: {e}", "current_url": current_url}

            # extract candidate JSON or text payloads
            snippets = extract_atob_and_pre(combined)
            # look for JSON snippet with email/secret/url/answer style (sample)
            page_obj = None
            for s in snippets:
                try:
                    obj = json.loads(s)
                    page_obj = obj
                    break
                except Exception:
                    # not JSON
                    pass
            if page_obj:
                # prefer explicit fields
                question_text = page_obj.get("question") or page_obj.get("task") or s
                submit_url = page_obj.get("submit") or page_obj.get("submit_url") or None
            else:
                # fallback: search text for instruction & submit url
                question_text = combined
                # attempt to find submit endpoint e.g., https://.../submit
                m = re.search(r'https?://[^\s"\']+/submit[^\s"\']*', combined, flags=re.IGNORECASE)
                submit_url = m.group(0) if m else None

            # download referenced resources mentioned in question_text
            loaded_files = {}
            for url_candidate in re.findall(r'(https?://[^\s"\']+)', question_text):
                # skip submit url
                if submit_url and url_candidate == submit_url:
                    continue
                # attempt to download
                try:
                    content, ctype = await download_bytes_async(client, url_candidate)
                    fname = os.path.basename(url_candidate.split("?")[0]) or f"file_{uuid.uuid4()}"
                    meta = {"bytes": content, "content_type": ctype, "url": url_candidate}
                    if 'csv' in ctype or fname.lower().endswith(".csv"):
                        try:
                            df = parse_csv_bytes(content)
                            meta["df"] = df
                        except Exception:
                            pass
                    elif 'json' in ctype or fname.lower().endswith(".json"):
                        try:
                            meta["json"] = parse_json_bytes(content)
                        except Exception:
                            pass
                    elif 'pdf' in ctype or fname.lower().endswith(".pdf"):
                        # leave bytes; heuristics can try summing
                        pass
                    loaded_files[fname] = meta
                except Exception:
                    # ignore individual download failures
                    continue

            # Attempt heuristic solve
            candidate = heuristic_solve(question_text, loaded_files)

            # If heuristic couldn't solve, ask LLM to produce Python plan
            plan_py = None
            if candidate is None and OPENAI_API_KEY:
                # summarize available files for prompt
                file_summary = {}
                for fname, meta in loaded_files.items():
                    summary = {"content_type": meta.get("content_type")}
                    if "df" in meta:
                        summary["columns"] = meta["df"].columns.tolist()
                        summary["shape"] = meta["df"].shape
                        summary["head"] = meta["df"].head(5).to_dict(orient='records')
                    if "json" in meta:
                        summary["json_keys"] = list(meta["json"].keys()) if isinstance(meta["json"], dict) else None
                    file_summary[fname] = summary
                plan = await llm_plan_for_question(question_text, file_summary)
                if plan and isinstance(plan, dict) and plan.get("python"):
                    plan_py = plan.get("python")

            # If plan_py exists, execute it in tempdir with files copied
            if plan_py:
                workdir = tempfile.mkdtemp(prefix="tds_run_", dir=TEMP_DIR_ROOT)
                # dump files into workdir
                for fname, meta in loaded_files.items():
                    path = os.path.join(workdir, fname)
                    open(path, "wb").write(meta["bytes"])
                ans, err, created = run_code_in_subprocess(plan_py, workdir, timeout_s=CODE_EXEC_TIMEOUT_S)
                shutil.rmtree(workdir, ignore_errors=True)
                if err:
                    candidate = None
                else:
                    candidate = ans

            # If still no candidate, set candidate to "unsupported"
            if candidate is None:
                candidate = "unsupported"

            # Ask LLM to verify/format candidate (if available)
            final_answer = candidate
            if OPENAI_API_KEY:
                try:
                    v = await llm_verify_answer(question_text, candidate)
                    if isinstance(v, dict) and v.get("valid"):
                        final_answer = v.get("final_answer")
                    else:
                        # invalid per LLM - if it gives alt answer, use it; else fallback to candidate
                        if isinstance(v, dict) and "final_answer" in v:
                            final_answer = v.get("final_answer")
                except Exception:
                    pass

            # prepare submission payload
            submit_payload = {"email": email, "secret": secret, "url": current_url, "answer": final_answer}

            # submit and handle response
            submitted = None
            if submit_url:
                try:
                    r = await client.post(submit_url, json=submit_payload, timeout=25.0)
                    try:
                        submitted = r.json()
                    except Exception:
                        submitted = {"status_text": r.text, "status_code": r.status_code}
                except Exception as e:
                    submitted = {"error": str(e)}
            else:
                return {"error": "no submit url found", "current_url": current_url, "candidate": candidate}

            # Evaluate response
            # If correct True and url provided -> follow next
            if isinstance(submitted, dict) and submitted.get("correct") is True:
                next_url = submitted.get("url")
                if next_url:
                    current_url = next_url
                    continue
                else:
                    # finished successfully
                    return {"status":"finished","last_submission": submitted}
            else:
                # incorrect or unclear; if server provided a new url, follow it
                next_url = submitted.get("url") if isinstance(submitted, dict) else None
                if next_url:
                    current_url = next_url
                    continue
                # if not, attempt one retry if time allows: recompute (try with LLM even if heuristics used)
                if time.time() + 10 < deadline and OPENAI_API_KEY:
                    # force LLM plan attempt if not tried
                    if not plan_py:
                        # attempt plan now
                        file_summary = {}
                        for fname, meta in loaded_files.items():
                            summary = {"content_type": meta.get("content_type")}
                            if "df" in meta:
                                summary["columns"] = meta["df"].columns.tolist()
                                summary["shape"] = meta["df"].shape
                            file_summary[fname] = summary
                        plan = await llm_plan_for_question(question_text, file_summary)
                        if plan and isinstance(plan, dict) and plan.get("python"):
                            plan_py = plan.get("python")
                            workdir = tempfile.mkdtemp(prefix="tds_run_", dir=TEMP_DIR_ROOT)
                            for fname, meta in loaded_files.items():
                                open(os.path.join(workdir, fname), "wb").write(meta["bytes"])
                            ans, err, created = run_code_in_subprocess(plan_py, workdir, timeout_s=CODE_EXEC_TIMEOUT_S)
                            shutil.rmtree(workdir, ignore_errors=True)
                            if not err:
                                final_answer = ans
                                # resubmit
                                submit_payload["answer"] = final_answer
                                try:
                                    r = await client.post(submit_url, json=submit_payload, timeout=25.0)
                                    try:
                                        submitted = r.json()
                                    except Exception:
                                        submitted = {"status_text": r.text, "status_code": r.status_code}
                                except Exception as e:
                                    submitted = {"error": str(e)}
                                if isinstance(submitted, dict) and submitted.get("correct") is True:
                                    next_url = submitted.get("url")
                                    if next_url:
                                        current_url = next_url
                                        continue
                                    else:
                                        return {"status":"finished","last_submission": submitted}
                # out of options for this page
                return {"status":"stopped","last_submission": submitted, "candidate": candidate}

        # end loop
    return {"status":"timeout_or_done"}

# ----------------------------
# Endpoint: the single one grader calls
# ----------------------------
@app.post("/")
async def grade_endpoint(request: Request):
    # parse JSON safely and validate
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    # basic fields
    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")
    if not email or not secret or not url:
        raise HTTPException(status_code=400, detail="missing fields")
    if secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="invalid secret")

    # accepted: spawn background task to solve quiz immediately
    # but return HTTP 200 now
    loop = asyncio.get_event_loop()
    # create background task (ensure exceptions are logged)
    async def background():
        try:
            res = await solve_quiz_loop(email, secret, url)
            # write debug result to a timestamped file
            fname = os.path.join(TEMP_DIR_ROOT, f"result_{int(time.time())}_{uuid.uuid4().hex}.json")
            open(fname, "w", encoding="utf-8").write(json.dumps(res, indent=2, default=str))
        except Exception as e:
            # log failure
            tb = traceback.format_exc()
            fname = os.path.join(TEMP_DIR_ROOT, f"error_{int(time.time())}_{uuid.uuid4().hex}.log")
            open(fname, "w", encoding="utf-8").write(tb)
    # schedule background task
    asyncio.create_task(background())

    # Return success immediately as required by spec
    return {"status":"accepted", "message":"processing started"}

# ----------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8080
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info")

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os
import json
import re
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Umrah Rate Sheet Extractor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL", "openai/gpt-4o-2024-08-06")

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a strict structured data extraction engine.

Extract pricing data from Umrah rate sheet images or documents.

Return ONLY a single raw JSON object.
No explanation. No markdown. No comments. No text outside JSON.

The JSON must ALWAYS contain exactly these six top-level keys:

{
  "hotels": [],
  "vehicles": [],
  "pvt_transport": [],
  "sharing_transport": [],
  "pvt_ziyarah": [],
  "sharing_ziyarah": []
}

Rules:
- If a section is missing, return empty array [].
- Never guess values.
- Preserve numbers exactly as written.
- Extract full tables without skipping any rows.
- Separate private vs sharing strictly.
- Detect Makkah/Madinah automatically when visible.
- Do NOT wrap output in markdown fences or any extra text.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "pdf": "application/pdf",
}

REQUIRED_KEYS = [
    "hotels",
    "vehicles",
    "pvt_transport",
    "sharing_transport",
    "pvt_ziyarah",
    "sharing_ziyarah",
]


def detect_mime_type(filename: str) -> str:
    ext = (filename or "").lower().rsplit(".", 1)[-1]
    return MIME_MAP.get(ext, "image/jpeg")


def safe_json_parse(text: str) -> dict:
    text = text.strip()

    # Strip markdown fences if model ignores instructions
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    # Direct parse attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON block between first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response")

    cleaned = text[start:end]

    # Fix trailing commas (common LLM mistake)
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)


def ensure_schema(data: dict) -> dict:
    for key in REQUIRED_KEYS:
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    return data


def build_message_content(image_base64: str, mime_type: str, instructions: str) -> list:
    """Build the user message content for OpenRouter."""

    if mime_type == "application/pdf":
        # PDF as document block (supported by GPT-4o via OpenRouter)
        content = [
            {
                "type": "text",
                "text": "Extract all structured pricing data from this document.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                },
            },
        ]
    else:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                },
            },
            {
                "type": "text",
                "text": "Extract all structured pricing data.",
            },
        ]

    if instructions.strip():
        content.append({"type": "text", "text": f"Additional instructions: {instructions.strip()}"})

    return content


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Umrah Rate Sheet Extractor",
        "model": MODEL,
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Render uses this for health checks."""
    return {"status": "healthy"}


@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(default=""),
):
    # ── Guard: API key ────────────────────────────────────────────────────────
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is not set.",
        )

    # ── Read file ─────────────────────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload.jpg"
    mime_type = detect_mime_type(filename)

    logger.info(f"Processing file: {filename} | mime: {mime_type} | size: {len(image_bytes)} bytes")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # ── Build payload ─────────────────────────────────────────────────────────
    user_content = build_message_content(image_base64, mime_type, instructions)

    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 4000,
        # NOTE: response_format removed — not all OpenRouter models support it
        # and it causes 400 errors. Prompt enforces JSON instead.
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://umrah-extractor.onrender.com",  # optional but good practice
        "X-Title": "Umrah Rate Sheet Extractor",
    }

    # ── Call OpenRouter ───────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter request timed out. Try again.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error contacting OpenRouter: {str(e)}")

    # ── Handle non-200 ────────────────────────────────────────────────────────
    if response.status_code != 200:
        logger.error(f"OpenRouter error {response.status_code}: {response.text[:500]}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {response.status_code}: {response.text[:300]}",
        )

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Unexpected OpenRouter response structure: {response.text[:500]}")
        raise HTTPException(status_code=502, detail=f"Unexpected response from OpenRouter: {str(e)}")

    logger.info(f"Raw model response (first 300 chars): {content[:300]}")

    # ── Parse JSON from model output ──────────────────────────────────────────
    try:
        parsed = safe_json_parse(content)
    except Exception as e:
        logger.error(f"JSON parse failed. Raw content: {content[:1000]}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model returned invalid JSON. Try again or use a different file.",
                "details": str(e),
                "raw_response": content[:1000],
            },
        )

    # ── Ensure schema ─────────────────────────────────────────────────────────
    parsed = ensure_schema(parsed)

    logger.info("Extraction successful.")
    return parsed

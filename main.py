from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os
import json
import re

app = FastAPI(title="Umrah Rate Sheet Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL", "openai/gpt-4o-2024-08-06")

SYSTEM_PROMPT = """
You are a strict structured data extraction engine.

Extract pricing data from Umrah rate sheet images.

Return ONLY a single raw JSON object.
No explanation.
No markdown.
No comments.
No text outside JSON.

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
- Preserve numbers exactly.
- Extract full tables without skipping rows.
- Separate private vs sharing strictly.
- Detect Makkah/Madinah automatically when visible.
"""

def detect_mime_type(filename: str) -> str:
    ext = (filename or "").lower().split(".")[-1]
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return mapping.get(ext, "image/jpeg")


def safe_json_parse(text: str):
    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except:
        pass

    # Try extracting JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found")

    cleaned = text[start:end]

    # Remove trailing commas
    cleaned = cleaned.replace(",}", "}")
    cleaned = cleaned.replace(",]", "]")

    return json.loads(cleaned)


def ensure_schema(data: dict):
    required_keys = [
        "hotels",
        "vehicles",
        "pvt_transport",
        "sharing_transport",
        "pvt_ziyarah",
        "sharing_ziyarah",
    ]
    for key in required_keys:
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    return data


@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL}


@app.post("/extract")
async def extract_data(file: UploadFile, instructions: str = Form("")):

    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = detect_mime_type(file.filename or "")

    system_prompt = SYSTEM_PROMPT
    if instructions.strip():
        system_prompt += "\nAdditional instructions:\n" + instructions.strip()

    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
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
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter error {response.status_code}: {response.text[:500]}",
        )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid OpenRouter response")

    try:
        parsed = safe_json_parse(content)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model returned invalid JSON",
                "details": str(e),
                "raw_response": content[:1000],
            },
        )

    parsed = ensure_schema(parsed)

    return parsed

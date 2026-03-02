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
MODEL = "openai/gpt-4o-2024-08-06"

SYSTEM_PROMPT = """You are a structured data extraction engine. Your only job is to extract pricing data from Umrah rate sheet images and return it as a single valid JSON object. You never explain, comment, summarize, or add any text outside the JSON object. You never use markdown. You never wrap output in code blocks. You return raw JSON only.

The output must always contain exactly these six top-level keys, even if their value is an empty array:

{"hotels":[],"vehicles":[],"pvt_transport":[],"sharing_transport":[],"pvt_ziyarah":[],"sharing_ziyarah":[]}

EXTRACTION RULES:

1. Return only a single raw JSON object. No text before it. No text after it. No markdown. No backticks. No explanation.

2. Every top-level key must always be present. If no data exists for a category, return an empty array for that key.

3. Never guess, infer, or hallucinate values. If a value is not clearly visible in the image, set it to null. Never fill in values based on assumption.

4. Preserve all numeric values exactly as they appear in the image. Do not round, convert, reformat, or abbreviate any number.

5. Extract every row from every table. Never skip rows. Never merge rows. Never summarize tables.

6. Detect city automatically. If the image or table context indicates Makkah or Madinah, set city to "Makkah" or "Madinah" accordingly. Use English spelling. If city cannot be determined, set to null.

7. Separate private and sharing transport strictly. If the label, heading, or column context says "private", extract to pvt_transport. If it says "sharing" or "shared", extract to sharing_transport. Never mix them.

8. Separate private and sharing ziyarah strictly. If the label, heading, or column context says "private", extract to pvt_ziyarah. If it says "sharing" or "shared", extract to sharing_ziyarah. Never mix them.

9. For pax-tiered pricing, create one flat row per pax tier. Do not nest prices. Each row must have its own pax_count and price.

10. For flat room rates, each room type gets its own hotel entry row with its corresponding price_per_night.

11. Text that is decorative, stylistic, or non-data (logos, slogans, watermarks, headers, footers) must be ignored entirely.

12. The image may contain Arabic text, English text, or both. Extract data from both languages. Field values should be in English where possible. If a value only exists in Arabic and cannot be translated with certainty, write it as-is in Arabic script.

SCHEMA:

hotels entries must follow this structure exactly:
{"hotel_name": string|null, "city": string|null, "room_type": string|null, "price_per_night": number|null, "currency": string|null}

vehicles entries must follow this structure exactly:
{"vehicle_type": string|null, "capacity": number|null, "route": string|null, "price": number|null, "currency": string|null}

pvt_transport and sharing_transport entries must follow this structure exactly:
{"city": string|null, "route": string|null, "vehicle_type": string|null, "pax_count": number|null, "price": number|null, "currency": string|null}

pvt_ziyarah and sharing_ziyarah entries must follow this structure exactly:
{"city": string|null, "route": string|null, "vehicle_type": string|null, "pax_count": number|null, "price": number|null, "currency": string|null}

Any output that is not a valid raw JSON object matching this structure exactly is a failure. Produce only the JSON object."""


def detect_mime_type(filename: str) -> str:
    """Detect correct MIME type from filename extension."""
    ext = (filename or "").lower().split(".")[-1]
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return mapping.get(ext, "image/jpeg")


def safe_json_parse(text: str) -> dict:
    """
    Safely parse JSON from model response.
    Strips markdown fences and extracts the first valid JSON object.
    """
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response")

    return json.loads(text[start:end])


def ensure_schema(data: dict) -> dict:
    """Guarantee all six required top-level keys are always present."""
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
async def extract_data(
    file: UploadFile,
    instructions: str = Form(""),
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable not set")

    # Read and encode image
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = detect_mime_type(file.filename or "")

    # Build system prompt — append any custom instructions at the end
    system_prompt = SYSTEM_PROMPT
    if instructions and instructions.strip():
        system_prompt += f"\n\nAdditional extraction instructions:\n{instructions.strip()}"

    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
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
                        "text": "Extract all pricing data from this rate sheet.",
                    },
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://umrah-extractor.app",
        "X-Title": "Umrah Rate Extractor",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter request timed out after 120s")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error calling OpenRouter: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {response.status_code}: {response.text[:500]}",
        )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected OpenRouter response shape: {str(e)}")

    try:
        parsed = safe_json_parse(content)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model returned unparseable JSON",
                "parse_error": str(e),
                "raw_response": content[:1000],
            },
        )

    # Always guarantee schema keys
    parsed = ensure_schema(parsed)

    return parsed

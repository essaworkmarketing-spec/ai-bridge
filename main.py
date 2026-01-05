from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import requests, base64, os

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "anthropic/claude-3.5-sonnet"

SYSTEM_PROMPT = """
You are a data extraction engine for Hajj & Umrah hotel rate sheets.

Your job:
- Extract ALL hotels from the image. Do NOT skip any.
- Detect CITY automatically (Makkah or Madinah etc).
- Extract:
  hotel_name
  city
  distance (if available)
  check_in_dates (if available)
  rates:
    sharing
    double
    triple
    quad
    quint
- If a rate is missing, return null.
- Do NOT guess values.
- Do NOT summarize.
- Output ONLY valid JSON.
- JSON must be an array of hotels.

Google Rating:
- If confident, include google_rating (number only).
- If unsure, set google_rating = null.
- Do NOT categorize or judge hotel quality.

IMPORTANT:
Return ALL hotels visible in the image, even if rates are missing.
"""

@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(default="")
):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode()

        final_prompt = SYSTEM_PROMPT
        if instructions.strip():
            final_prompt += f"\n\nExtra instructions:\n{instructions}"

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_base64}"
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 2500
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )

        data = res.json()

        if "error" in data:
            return JSONResponse(status_code=500, content=data)

        raw_text = data["choices"][0]["message"]["content"]

        return {
            "status": "success",
            "hotels": raw_text
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

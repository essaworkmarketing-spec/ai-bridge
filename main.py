from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import os, base64, json, re, requests
from google.cloud import vision

app = FastAPI()

# ---------------- CONFIG ----------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CLAUDE_MODEL = "anthropic/claude-3.5-haiku"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ---------------- OCR ----------------
def run_ocr(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    return response.full_text_annotation.text

# ---------------- AI FORMATTER ----------------
def format_with_ai(ocr_text):
    prompt = f"""
You are a data formatter.

Input is OCR text from hotel rate sheets.

Rules:
- Do NOT guess
- Extract ALL hotels
- Detect city from headings (MAKKAH / MADINAH)
- Keep hotel even if rates missing
- Missing rates = 0
- Output STRICT JSON only
- No explanations

JSON schema:
{{
  "hotels": [
    {{
      "name": "",
      "city": "",
      "rates": {{
        "sharing": 0,
        "quint": 0,
        "quad": 0,
        "triple": 0,
        "double": 0
      }}
    }}
  ],
  "transport": [],
  "ziyarat": []
}}
TEXT:
{ocr_text}
"""

    payload = {
        "model": CLAUDE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1200
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    data = r.json()

    raw = data["choices"][0]["message"]["content"]

    # Safe JSON extraction
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise Exception("AI did not return JSON")

    return json.loads(match.group())

# ---------------- NORMALIZER (LOVABLE SAFE) ----------------
def normalize(data):
    hotels = []
    for h in data.get("hotels", []):
        rates = h.get("rates", {})
        clean_rates = {}
        has_rate = False

        for k in ["sharing", "quint", "quad", "triple", "double"]:
            v = rates.get(k, 0)
            try:
                num = int(re.findall(r"\d+", str(v))[0])
            except:
                num = 0
            if num > 0:
                has_rate = True
            clean_rates[k] = num

        hotels.append({
            "name": h.get("name", "").strip(),
            "city": h.get("city", ""),
            "rates": clean_rates,
            "has_any_rate": has_rate
        })

    return {
        "hotels": hotels,
        "transport": data.get("transport", []),
        "ziyarat": data.get("ziyarat", [])
    }

# ---------------- API ----------------
@app.post("/extract")
async def extract(file: UploadFile):
    try:
        image_bytes = await file.read()

        # 1. OCR
        ocr_text = run_ocr(image_bytes)

        # 2. AI formatting
        structured = format_with_ai(ocr_text)

        # 3. Normalize for Lovable
        final_output = normalize(structured)

        # Safety check
        if len(final_output["hotels"]) == 0:
            raise Exception("No hotels detected")

        return final_output

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

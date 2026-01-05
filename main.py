from fastapi import FastAPI, UploadFile, Form
import requests
import base64
import os
import json

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-3.5-sonnet"

SYSTEM_PROMPT = """
You are a strict JSON generator.

CRITICAL RULES:
- Output ONLY valid JSON
- Use double quotes for ALL strings
- Escape special characters
- No line breaks inside values
- No comments
- No trailing commas
- Text like Coming Soon or Flat Room Rate MUST be strings

JSON FORMAT (MANDATORY):

{
  "hotels": [
    {
      "name": "string",
      "city": "Makkah | Madinah | Unknown",
      "rates": {
        "sharing": "string",
        "quint": "string",
        "quad": "string",
        "triple": "string",
        "double": "string"
      }
    }
  ],
  "transport": [],
  "ziyarat": []
}

If data missing, use "N/A".
Return ONLY JSON. Nothing else.
"""

def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}") + 1
        cleaned = text[start:end]
        return json.loads(cleaned)

@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form("")
):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    user_prompt = f"""
Extract ALL data from this rate sheet image.
Hotels, cities, room prices, missing values, coming soon.
{instructions}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}"
                    }
                ]
            }
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        return {
            "error": "OpenRouter API failed",
            "details": response.text
        }

    data = response.json()
    content = data["choices"][0]["message"]["content"]

    parsed = safe_json_parse(content)

    return parsed

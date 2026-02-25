from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "openai/gpt-4o-2024-08-06"

SYSTEM_PROMPT = """
You are a strict data extraction engine.

Extract ALL structured pricing data from the uploaded rate sheet image.

Return ONLY valid JSON in this format:

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

Rules:
- No explanation text
- No markdown
- No comments
- If missing value use "N/A"
- Do not guess values
- Preserve numeric accuracy exactly
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
async def extract_data(file: UploadFile, instructions: str = Form("")):
    try:
        # Read image once
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Optional instruction append
        system_prompt = SYSTEM_PROMPT
        if instructions:
            system_prompt += "\nExtra instructions:\n" + instructions

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all structured pricing data from this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2500,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:
            return {
                "error": "OpenRouter API failed",
                "details": response.text
            }

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        parsed = safe_json_parse(content)

        return parsed

    except Exception as e:
        return {"error": str(e)}

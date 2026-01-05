from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os
import json

app = FastAPI()

# CORS (Lovable / Frontend ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-3.5-sonnet"


@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(None)
):
    try:
        # 1. Read image
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 2. System prompt (STRICT JSON)
        system_prompt = """
You are a data extraction engine.

Rules:
- Extract EVERYTHING from the image
- Hotels, cities (Makkah / Madinah), room rates
- Transport, ziyarat, car types if present
- Detect city automatically
- If data missing write "N/A"
- Return VALID JSON ONLY
- No explanation text
- No markdown
"""

        if instructions:
            system_prompt += "\nExtra instructions:\n" + instructions

        # 3. Claude payload
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
                            "text": "Extract all structured data from this rate sheet."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_base64}"
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 2000
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        # 4. Call OpenRouter
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        result = response.json()

        # 5. Extract Claude JSON safely
        content = result["choices"][0]["message"]["content"]

        # Claude sometimes returns string JSON
        if isinstance(content, str):
            content = content.strip()
            parsed = json.loads(content)
        else:
            parsed = content

        return parsed

    except Exception as e:
        return {
            "error": str(e)
        }

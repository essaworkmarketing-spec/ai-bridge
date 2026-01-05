from fastapi import FastAPI, UploadFile, Form
import requests
import base64
import os
import json

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
You are a professional data extraction engine for Hajj & Umrah rate sheets.

Your task:
Extract ALL data from the file without skipping anything.

You MUST extract:
- Hotels
- Ziyarat packages
- Transport services
- Vehicle / car types
- Routes & distances
- Rates (sharing, double, triple, quad, quint, flat)
- Check-in dates
- Notes (rate reduced, coming soon, flat rate)
- City (infer from section headers like MAKKAH HOTELS, MADINAH HOTELS)

Rules:
- Do not assume hotel-only data
- Do not stop after first category
- Classify each row correctly
- If value missing, return null
- Return structured JSON ONLY
- No explanations
"""

@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(default="")
):
    try:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        final_prompt = SYSTEM_PROMPT
        if instructions.strip():
            final_prompt += "\n\nAdditional user instructions:\n" + instructions

        payload = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded}"
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1200
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )

        data = response.json()

        # HARD FAIL if model returns error
        if "error" in data:
            return {
                "status": "error",
                "message": data["error"]
            }

        # Extract only the content text
        raw_text = data["choices"][0]["message"]["content"]

        # Claude usually returns clean JSON, but we still protect parsing
        try:
            extracted_json = json.loads(raw_text)
        except:
            extracted_json = {
                "raw_output": raw_text
            }

        return {
            "status": "success",
            "data": extracted_json
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

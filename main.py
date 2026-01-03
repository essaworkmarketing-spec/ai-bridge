from fastapi import FastAPI, UploadFile, HTTPException
import requests
import base64
import os
import json

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen2.5-vl-7b-instruct:free"

SYSTEM_PROMPT = """
You are a data extraction engine.
Return ONLY valid JSON.
No text, no explanation.

Required JSON structure:

{
  "hotels": [],
  "transport": [],
  "ziyarat": [],
  "routes": [],
  "visa": []
}

If any section is missing in the image, return an empty array for it.
"""

@app.post("/extract")
async def extract_data(file: UploadFile):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    content = await file.read()
    encoded_image = base64.b64encode(content).decode("utf-8")

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all hotel, transport, ziyarat, visa, and route rate data from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{encoded_image}"
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

    response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)

    data = response.json()

    # 🚨 Detect OpenRouter error
    if "error" in data:
        raise HTTPException(status_code=400, detail=data["error"])

    try:
        # Extract AI text
        ai_text = data["choices"][0]["message"]["content"]

        # Parse JSON safely
        extracted_json = json.loads(ai_text)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI response parsing failed: {str(e)}"
        )

    # ✅ GUARANTEE required keys for Lovable
    final_output = {
        "hotels": extracted_json.get("hotels", []),
        "transport": extracted_json.get("transport", []),
        "ziyarat": extracted_json.get("ziyarat", []),
        "routes": extracted_json.get("routes", []),
        "visa": extracted_json.get("visa", [])
    }

    return final_output

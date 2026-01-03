from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import base64
import os

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen-2.5-vl-7b-instruct:free"

@app.post("/extract")
async def extract_data(
    file: UploadFile = File(...),
    prompt: str = "Extract hotel, transport, visa, ziyarat, routes, and rates data. Return STRICT JSON only."
):
    try:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded}"
                        }
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",
            "X-Title": "Packitfy AI Bridge"
        }

        res = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)

        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=res.text)

        data = res.json()

        # 🚨 OpenRouter error handling
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"])

        # ✅ CLEAN JSON ONLY (Lovable expects this)
        content = data["choices"][0]["message"]["content"]

        return {
            "success": True,
            "data": content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, UploadFile, File
import requests
import base64
import os

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    # Read file
    content = await file.read()
    encoded = base64.b64encode(content).decode("utf-8")

    payload = {
        # ✅ CORRECT MODEL ID (NO :free)
        "model": "qwen/qwen-2.5-vl-7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract hotel, transport, visa, ziyarat, routes, "
                            "hotel names, categories, ratings, and all rates. "
                            "Return STRICT JSON only."
                        )
                    },
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
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    return response.json()

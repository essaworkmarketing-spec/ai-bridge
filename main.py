from fastapi import FastAPI, UploadFile
import requests
import base64
import os

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.post("/extract")
async def extract_data(file: UploadFile):
    content = await file.read()
    encoded = base64.b64encode(content).decode("utf-8")

    payload = {
        "model": "qwen/qwen2.5-vl-7b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract hotel, transport, visa, ziyarat, routes, and rates data. Return JSON only."
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
        json=payload,
        headers=headers
    )

    return response.json()

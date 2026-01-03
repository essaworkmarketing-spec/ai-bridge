from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import base64
import os

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY missing")

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
                        "text": "Extract hotel, transport, visa, ziyarat, routes, and rates. Return clean JSON only."
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
        headers=headers,
        timeout=60
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=response.text
        )

    data = response.json()

    if "choices" not in data:
        raise HTTPException(status_code=500, detail=data)

    return data["choices"][0]["message"]["content"]

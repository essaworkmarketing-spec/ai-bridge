from fastapi import FastAPI, UploadFile, HTTPException
import requests
import base64
import os

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

@app.post("/extract")
async def extract_data(file: UploadFile):
    try:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        payload = {
            "model": "qwen/qwen2.5-vl-7b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract hotel, transport, visa, ziyarat, routes, and rates. Return valid JSON only."
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
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourapp.com",
            "X-Title": "Flyer AI Extractor"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )

        data = response.json()

        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"])

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

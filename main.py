from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import requests, base64, os, re

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
You are a data extraction engine.

Extract ALL hotels, transport and ziyarat data.
Do NOT format as markdown.
Do NOT add explanations.

Return data in plain structured text like:

HOTEL:
Name: ARAFAT GOLDEN
City: Makkah
Sharing: 16
Quint: 20
Quad: 32
Triple: 24
Double: 48

Repeat for ALL rows.
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
            final_prompt += "\nUser Instructions:\n" + instructions

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
            "temperature": 0,
            "max_tokens": 1800
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )

        result = response.json()
        text = result["choices"][0]["message"]["content"]

        # -------- SAFE PARSING --------
        hotels = []
        blocks = re.split(r"\n\s*HOTEL:\s*", text)

        for block in blocks[1:]:
            def find(label):
                m = re.search(label + r":\s*(.+)", block)
                return m.group(1).strip() if m else None

            hotels.append({
                "name": find("Name"),
                "city": find("City"),
                "rates": {
                    "sharing": find("Sharing"),
                    "quint": find("Quint"),
                    "quad": find("Quad"),
                    "triple": find("Triple"),
                    "double": find("Double")
                }
            })

        return {
            "hotels": hotels,
            "transport": [],
            "ziyarat": []
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

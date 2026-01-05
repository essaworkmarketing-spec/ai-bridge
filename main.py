from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import requests, base64, os, re, json

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SYSTEM_PROMPT = """
You are a professional data extraction engine for Hajj & Umrah travel agencies.

Rules:
1. Extract ALL entities present in the file. No limits.
2. Detect automatically:
   - Hotels
   - City (Makkah / Madinah / Other if mentioned)
   - Room rates (sharing, double, triple, quad, quint, etc.)
   - Transport (shuttle, private, car types)
   - Ziyarat packages
3. Do NOT summarize.
4. Do NOT skip rows.
5. If a value is missing, return null.
6. Return ONLY valid JSON in the exact schema below.
7. Never add explanations or text outside JSON.

Schema:
{
  "hotels": [
    {
      "name": "",
      "city": "",
      "rates": {
        "sharing": null,
        "double": null,
        "triple": null,
        "quad": null,
        "quint": null
      }
    }
  ],
  "transport": [],
  "ziyarat": []
}
"""

@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(default="")
):
    try:
        # Read & encode image/file
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")

        user_prompt = SYSTEM_PROMPT
        if instructions.strip():
            user_prompt += f"\nAdditional instructions:\n{instructions}"

        payload = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded}"
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

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )

        result = response.json()

        if "choices" not in result:
            return JSONResponse(
                status_code=500,
                content={"error": result}
            )

        raw_text = result["choices"][0]["message"]["content"]

        # ---- HARD JSON EXTRACTION ----
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            return JSONResponse(
                status_code=500,
                content={"error": "No JSON detected from AI"}
            )

        clean_json = json.loads(json_match.group())

        # ---- FINAL NORMALIZATION FOR LOVABLE ----
        final_output = {
            "hotels": clean_json.get("hotels", []),
            "transport": clean_json.get("transport", []),
            "ziyarat": clean_json.get("ziyarat", [])
        }

        return final_output

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

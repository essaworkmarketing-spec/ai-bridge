from fastapi import FastAPI, UploadFile, Form, HTTPException
import requests
import base64
import os
import json

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "anthropic/claude-3.5-sonnet"

BASE_SYSTEM_PROMPT = """
You are a professional data extraction engine.

CORE RULES (NON-NEGOTIABLE):
- Extract ALL data visible in the image
- NEVER summarize
- NEVER limit results
- NEVER skip entries
- If 1 hotel exists → extract it
- If 100 hotels exist → extract all 100
- If any value is missing → return null, do NOT skip

You must return STRICT VALID JSON only.
No explanations. No markdown. No comments.

BASE OUTPUT FORMAT:

{
  "hotels": [
    {
      "hotel_name": "",
      "distance": "",
      "google_rating": null,
      "check_in": "",
      "transport": "",
      "room_rates": {
        "single": null,
        "double": null,
        "triple": null,
        "quad": null,
        "sharing": null
      },
      "raw_text": ""
    }
  ]
}

GOOGLE RATING RULES:
- Fetch approximate Google rating for each hotel
- Rating must be a number like 4.1
- If not found, return null
- DO NOT categorize or filter by rating
"""

@app.post("/extract")
async def extract_data(
    file: UploadFile,
    custom_instruction: str = Form(default="")
):
    try:
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Merge system instructions
        final_system_prompt = BASE_SYSTEM_PROMPT

        if custom_instruction.strip():
            final_system_prompt += f"""

CUSTOM USER INSTRUCTIONS (HIGH PRIORITY):
{custom_instruction}

Follow these instructions strictly while still obeying CORE RULES.
"""

        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "system",
                    "content": final_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract complete structured data from this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{image_base64}"
                        }
                    ]
                }
            ],
            "temperature": 0,
            "max_tokens": 4500
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.local",
            "X-Title": "Unlimited Rate Extractor"
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=response.text
            )

        ai_response = response.json()

        if "choices" not in ai_response:
            raise HTTPException(
                status_code=500,
                detail="Invalid AI response structure"
            )

        content = ai_response["choices"][0]["message"]["content"]

        # Ensure valid JSON
        try:
            parsed = json.loads(content)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="AI did not return valid JSON"
            )

        if "hotels" not in parsed or not isinstance(parsed["hotels"], list):
            raise HTTPException(
                status_code=500,
                detail="Hotels array missing in AI response"
            )

        return parsed

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

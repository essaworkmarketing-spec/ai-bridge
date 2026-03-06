# Umrah Rate Sheet Extractor API

FastAPI backend that extracts structured pricing data from Umrah rate sheet images using OpenRouter AI.

## Files

```
main.py            ← FastAPI app
requirements.txt   ← Python dependencies
Procfile           ← Start command for Render
render.yaml        ← Render deployment config (optional)
```

## Deploy on Render

1. Push all files to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment:** Python 3
5. Add Environment Variable:
   - `OPENROUTER_API_KEY` → your key from openrouter.ai
   - `MODEL` → `openai/gpt-4o-2024-08-06` (optional, this is the default)

## API Endpoints

| Method | Endpoint   | Description               |
|--------|------------|---------------------------|
| GET    | `/`        | Status check              |
| GET    | `/health`  | Health check for Render   |
| POST   | `/extract` | Upload rate sheet & extract |

## POST /extract

**Form Data:**
- `file` — image file (jpg, png, webp, gif) or PDF
- `instructions` — (optional) extra instructions for the model

**Response:**
```json
{
  "hotels": [],
  "vehicles": [],
  "pvt_transport": [],
  "sharing_transport": [],
  "pvt_ziyarah": [],
  "sharing_ziyarah": []
}
```

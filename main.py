from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os
import json
import re
import logging
import io

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Umrah Rate Sheet Extractor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.getenv("MODEL", "google/gemini-2.5-flash")

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a strict structured data extraction engine for Umrah/Hajj rate sheets.

You will receive data in ANY format: image, PDF, plain text, or Excel (converted to text). Extract from all formats equally.

Return ONLY a single raw JSON object. No explanation. No markdown. No comments. No text outside JSON.

The JSON must ALWAYS contain exactly these seven top-level keys:
{
  "hotels": [],
  "vehicles": [],
  "pvt_transport": [],
  "sharing_transport": [],
  "pvt_ziyarah": [],
  "sharing_ziyarah": [],
  "rate_sheets": []
}

════════════════════════════════════════
CUSTOM INSTRUCTIONS:
════════════════════════════════════════
If the user provides additional instructions, follow them strictly.
Custom instructions override default behavior where they conflict.
Apply them during extraction, not after.

════════════════════════════════════════
HOTELS EXTRACTION RULES:
════════════════════════════════════════
Each hotel object:
{
  "city": "Makkah" or "Madinah",
  "name": "full hotel name as written",
  "location": "location/area as written",
  "distance": "distance value as written e.g. 700-800 MTR or SHUTTLE SERVICE",
  "sharing": null or number,
  "quint": null or number,
  "quad": null or number,
  "triple": null or number,
  "double": null or number,
  "flat_room_rate": null or number,
  "confidence": "high" or "low"
}

- Extract EVERY row from BOTH Makkah and Madinah hotel tables.
- CRITICAL: If hotel is under "MAKKAH HOTELS" header set city = "Makkah". If under "MADINAH HOTELS" header set city = "Madinah". Never leave city blank or unknown.
- If a rate cell says "N/A" or is blank, use null.
- Preserve exact numbers. Never skip any row or rate column.

CONFIDENCE FLAG — be honest, this powers a human review step:
- Set "confidence": "high" ONLY when you can read the row's numbers clearly and you
  are sure they are correct.
- Set "confidence": "low" whenever ANY of these is true for that row:
    • a number is blurry, cut off, or hard to read
    • the row shares its distance/rate with the rows above it and you are not fully
      sure the number belongs to THIS row (copy-down risk)
    • a merged cell makes it unclear which column a number belongs to
    • the hotel name spans multiple lines or is hard to separate from the next row
    • you had to guess anything at all
- It is BETTER to mark "low" and be safe than to mark "high" and be wrong. A human
  will double-check every "low" row. Do not hide uncertainty behind "high".

────────────────────────────────────────
FLAT ROOM RATE — READ THIS CAREFULLY:
────────────────────────────────────────
The phrase "FLAT ROOM RATE" may appear in TWO different table layouts. Handle BOTH.

LAYOUT A — flat rate sits inside a specific room-type column
  Example: the row has columns Sharing | Quint | Quad | Triple | Double
  and the text "FLAT ROOM RATE 490/-" is written across the Triple/Double cells.
  → Set flat_room_rate = 490. Set sharing, quint, quad, triple, double = null.

LAYOUT B — one wide merged cell under a "ROOM RATE" group header
  Example: there is a group header "ROOM RATE" with sub-columns Quint | Quad | Triple | Double,
  and most rows have a single merged cell spanning that whole block that reads
  "FLAT ROOM RATE 70", "FLAT ROOM RATE 185", "FLAT ROOM RATE 700/-" etc.
  → That number is the flat_room_rate. Set flat_room_rate = 70 (or 185, 700...).
    Set sharing, quint, quad, triple, double = null for that row.

MIXED ROWS (very important):
  In Layout B some rows put a real number in ONE sub-column (e.g. Quad = 170, or Triple = 225)
  AND still show "FLAT ROOM RATE 150" for the rest of the block.
  Example row: "FAJAR BADEA 4 ... Quad column = 170 ... FLAT ROOM RATE 150"
  → Put 170 in quad. Put 150 in flat_room_rate. Keep both. Do not drop either one.
  Example row: "EMAAR AL KHALIL ... Quad 225 ... Triple 210 ... Double 195"
  → This row has NO flat text, so fill quad=225, triple=210, double=195, flat_room_rate=null.

HARD RULES for flat room rate:
- ANY time you see the words "FLAT ROOM RATE" (in any casing, with or without /-, with or without a colon)
  followed by a number ANYWHERE in a hotel row, that number MUST go into flat_room_rate.
- Never leave flat_room_rate null if the row visibly contains the words "FLAT ROOM RATE" and a number.
- Never mistake "FLAT ROOM RATE 70" for a distance, a column header, or plain text. It is always a price.
- Extract the number only. "FLAT ROOM RATE 700/-" → 700. "FLAT ROOM RATE 185" → 185.
- A table can have only Quint/Quad/Triple/Double columns and NO Sharing column. That is normal. Leave sharing=null.

NEVER DUPLICATE THE FLAT RATE INTO A COLUMN — READ THIS:
- The flat rate number belongs ONLY in flat_room_rate. It does NOT belong in double,
  triple, quad, quint, or sharing.
- The "FLAT ROOM RATE 70" cell is one wide merged cell. Its right edge happens to sit
  under the Double column, but that does NOT mean Double = 70. Double is EMPTY on that row.
- WRONG: flat_room_rate = 70 AND double = 70 (you copied the same number twice).
- RIGHT: flat_room_rate = 70, double = null, triple = null, quad = null, quint = null, sharing = null.
- Only put a number in double/triple/quad/quint when that specific column shows its OWN
  separate number that is DIFFERENT from the flat rate (like FAJAR BADEA 4: quad 170, or
  EMAAR AL KHALIL: quint 225, triple 210, double 195 with no flat text at all).
- If a row has a flat rate AND every per-type column would just repeat that same number,
  the per-type columns are all null. A number never appears in two fields on the same row
  unless the sheet literally prints it in two different column cells.

════════════════════════════════════════
PRIVATE TRANSPORT EXTRACTION RULES:
════════════════════════════════════════
pvt_transport = per vehicle pricing. Input is usually a table with vehicle columns.

Each object:
{
  "route": "route name as written",
  "rates": {
    "<vehicle column header exactly as printed in the sheet>": number
  }
}

- Include EVERY route row.
- Read the vehicle column headers from THIS sheet only. Do NOT use any predefined,
  remembered, or example vehicle list.
- Use each column header exactly as printed, including slashes and the word "or".
  Valid keys look like: "CAR", "STAREX OR STARIA", "GMC", "HIACE", "COASTER",
  "H1/HYUNDAI", "GRAND CABIN". Whatever this sheet prints is what you output.
- Add one key per vehicle column that has a price in that row.
- If a cell is blank, dashed, or unreadable, OMIT that key entirely.
  Never write null and never guess a number.
- Never output a vehicle column that is not printed in this sheet.

════════════════════════════════════════
SHARING TRANSPORT EXTRACTION RULES:
════════════════════════════════════════
sharing_transport = per person transport pricing between cities/routes.

Input can be ANY of these formats:
  FORMAT A (table): route rows with per_person price column
  FORMAT B (plain text bullet list):
    - Jeddah to Makkah 70sar
    - Makkah to Madinah 70 sar
  FORMAT C (labeled plain text):
    sharing transport rates
    - Jeddah to Makkah 70sar
    - Makkah to Jeddah 70 sar

Each object:
{
  "route": "route name e.g. Jeddah to Makkah",
  "per_person": number,
  "currency": "SAR"
}

- Parse ALL formats above equally.
- Strip sar/SAR/SR suffix, extract only the number into per_person.
- Extract EVERY route without skipping.
- If a header says "sharing transport" treat ALL items below it as sharing_transport.

════════════════════════════════════════
SHARING ZIYARAH EXTRACTION RULES:
════════════════════════════════════════
sharing_ziyarah = per person ziyarah tour pricing.

Input can be ANY of these formats:
  FORMAT A (table): tour rows with per_person price column
  FORMAT B (single line per tour):
    Badar Ziyarat SAR 70/person
    Madinah Ziyarat SAR 70/person
  FORMAT C (alternating lines):
    Badar Ziyarat
    SAR 70/person
    Madinah Ziyarat
    SAR 70/person

Each object:
{
  "tour_name": "name as written",
  "city": "Makkah" or "Madinah",
  "per_person": number,
  "currency": "SAR"
}

City detection rules:
- Badar Ziyarat = "Madinah"
- Madinah Ziyarat = "Madinah"
- Makkah Ziyarat = "Makkah"
- Taif Ziyarat = "Makkah"
- Any tour with "Madinah" in name = "Madinah"
- Any tour with "Makkah" or "Taif" in name = "Makkah"

- Parse ALL formats above equally.
- Strip SAR/ prefix and /person suffix, extract only the number.
- Extract EVERY tour without skipping.

════════════════════════════════════════
PRIVATE ZIYARAH EXTRACTION RULES:
════════════════════════════════════════
pvt_ziyarah = per vehicle ziyarah pricing.

Each object:
{
  "city": "Makkah" or "Madinah",
  "tour_name": "name as written",
  "rates": {
    "<vehicle column header exactly as printed in the sheet>": number
  },
  "per_person": null or number
}

- Read the vehicle column headers from THIS sheet only. Do NOT use any predefined,
  remembered, or example vehicle list.
- Use each column header exactly as printed. OMIT any vehicle whose cell is blank,
  dashed, or unreadable. Never write null and never guess a number.
- Never output a vehicle column that is not printed in this sheet.

════════════════════════════════════════
VEHICLES EXTRACTION RULES:
════════════════════════════════════════
Return one object for EVERY distinct vehicle column or vehicle name that appears
anywhere in this sheet, even when there is no standalone capacity table:
{
  "name": "vehicle name exactly as printed in the column header",
  "type": "vehicle type if stated, else null",
  "capacity": null or number,
  "description": "any notes"
}

- Read capacity from the vehicle column header when it is printed there:
  "CAR 3 SEATER" gives capacity 3.
  "STAREX OR STARIA 7 SEATER" gives capacity 7.
  "HIACE 9 SEATER" gives capacity 9.
  "COASTER 15-16 SEATER" gives capacity 16 (for a range, use the higher number).
- Never rename or normalize a vehicle. Keep the header text exactly as printed.
- Never add a vehicle that is not printed in this sheet.

════════════════════════════════════════
GLOBAL RULES:
════════════════════════════════════════
- Never guess or invent values.
- Never map a vehicle onto a predefined or remembered vehicle name. Vehicle names
  ALWAYS come from this sheet's own column headers, exactly as printed.
- Preserve all numbers exactly as written.
- Extract EVERY row from EVERY table and EVERY item from EVERY list — zero skipping.
- If a section is not present in the input, return empty array [].
- Handle image, PDF, plain text, and Excel (as text) inputs equally well.
- Do NOT wrap output in markdown fences or any extra text.
- Output must be valid parseable JSON only.

════════════════════════════════════════
ACCURACY — READ EVERY ROW INDEPENDENTLY:
════════════════════════════════════════
This is critical. Long tables cause a specific failure: the model reads the
first few rows correctly, then starts copying the previous row's number down
the rest of the column instead of reading each cell fresh. DO NOT do this.

- Read the number in EACH row separately, with fresh eyes, as if it were the
  only row on the page. Never carry a value down from the row above.
- Two rows next to each other often have DIFFERENT numbers even when they look
  similar (e.g. 250, 260, 270, 300, 320, 340, 360, 370 in consecutive rows).
  Read each one. Do not flatten them all to the same value.
- Before finishing, re-scan the last third of every long table. That is where
  copy-down errors happen most. Verify each flat_room_rate against the row it
  belongs to.
- If a row has a real number in a specific column (Quint, Quad, Triple, Double)
  AND a flat rate, capture BOTH. Do not drop the column number in favor of the
  flat rate, and do not drop the flat rate in favor of the column number.
- Count the rows in each table. Your output must contain exactly that many hotel
  objects for that city. If a table has 17 visible hotel rows, return 17 — never
  16, never skip a row because it looks like the one above it.

════════════════════════════════════════
SINGLE HOTEL RATE SHEETS (the "rate_sheets" key):
════════════════════════════════════════
Some files are not a list of many hotels. They are ONE hotel's own rate card,
with the hotel's rooms across the top and dated periods down the side, often
split into floor bands (Regular Floor, Premium Floor, Executive Floor) and with
an extra bed, parking and a meals table underneath.

When the file is that kind of sheet, fill "rate_sheets". Otherwise leave it [].

Each entry:
{
  "hotel": "Dallah Taiba Hotel",
  "city": "Makkah" or "Madinah",
  "distance": "distance from the Haram as written, or null",
  "year_hint": "2026-27",
  "floors": [
    {
      "floor": "regular" | "premium" | "executive" | null,
      "covers": "Floors 2 to 8",
      "extra_bed": 60 or null,
      "rooms": ["double", "triple", "quad", "junior_suite_1"],
      "room_labels": ["Twin / Double", "Triple", "Quad", "Junior Suite 1 (2 Room W/1 Washroom)"],
      "room_details": ["", "", "", "2 rooms with 1 washroom"],
      "pax": [2, 3, 4, 4],
      "periods": [
        { "from": "2026-07-01", "to": "2026-08-20", "rates": [513, 574, 634, 755] },
        { "from": "2026-08-20", "to": "2026-09-01", "rates": [574, 634, 694, 815] }
      ]
    }
  ],
  "extras": [
    { "type": "breakfast", "floor": null, "rate": 60 },
    { "type": "breakfast", "floor": "executive", "rate": 70 },
    { "type": "parking", "floor": null, "rate": 60 }
  ]
}

RULES FOR rate_sheets:
- "rooms" uses ONLY these keys, in the order the columns appear on the sheet:
  sharing, quint, quad, triple, double, flat_room_rate, junior_suite_1,
  junior_suite_2, senior_suite_2, senior_suite_3, apartment, executive_suite,
  diplomatic_suite, premium_suite, royal_suite.
  "Twin / Double" and "All Executive Twin / Double" are both "double".
  "Senoir Suite 2" is a misspelling of Senior Suite 2 and maps to "senior_suite_2".
  "Executive / Business Suites" maps to "executive_suite".
  A column you cannot map to one of these keys is skipped entirely: drop it from
  "rooms", "room_labels", "room_details", "pax" AND from every period's "rates".
- "room_labels" is the column header exactly as printed.
- "room_details" is only the part in brackets that describes the room, rewritten
  plainly: "(2 Room W/1 Washroom)" becomes "2 rooms with 1 washroom". Empty
  string when the header says nothing about rooms or washrooms.
- "pax" is the PAX number printed under the column header.
- Every period's "rates" array MUST have exactly the same length as "rooms", in
  the same order. Use null for a blank or N/A cell. Never shift the row left.
- "from" and "to" are ISO dates, YYYY-MM-DD. The sheet usually prints them as
  "01-Jul  20-Aug" with the season year in the title ("Rates 1448 | 2026-27").
  Use the first year from the title for the earliest period, and roll to the
  next year as soon as the months wrap around: 01-Jul-2026 to 20-Aug-2026, then
  20-Aug-2026 to 01-Sep-2026, and a period reading "20-Dec 10-Jan" becomes
  2026-12-20 to 2027-01-10.
- "floor" is one of regular, premium, executive, matched from the band heading
  ("Regular Floor (02 Till 8th Floor)" is "regular"). A sheet with no floor
  bands uses null and a single entry in "floors".
- "covers" is the floor range in plain words: "Floors 2 to 8", "Floors 9 to 11".
- "extra_bed" is that floor's extra bed price per night, from the table under
  the rates ("Premium Floor - Extra Bed ... SAR 60/- Per Night" gives 60 on the
  premium floor). A floor with no extra bed row gets null. NEVER copy another
  floor's extra bed price onto a floor that does not list one.
- "extras" covers the meals and parking tables. "type" is one of breakfast,
  lunch, dinner, half_board, full_board, parking, other. "floor" is null when
  the price applies to every floor, or the floor band it is listed against.
  A meals table with a "Regular" row and an "Executive" row means: the Regular
  prices go in with floor null, and the Executive prices go in with floor
  "executive". Parking is per car per night.
- Read every period row of every floor. A sheet with 3 floors and 6 periods has
  18 period objects in total. Do not stop after the first floor.
- When you fill "rate_sheets", still fill "hotels" with one entry for this hotel
  carrying its cheapest standard room rates, so older screens keep working.
"""

# ── File type helpers ─────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif"}
TEXT_EXTENSIONS = {"txt", "csv", "text"}
PDF_EXTENSIONS = {"pdf"}
EXCEL_EXTENSIONS = {"xlsx", "xls"}

MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "pdf": "application/pdf",
}

REQUIRED_KEYS = [
    "hotels",
    "vehicles",
    "pvt_transport",
    "sharing_transport",
    "pvt_ziyarah",
    "sharing_ziyarah",
    "rate_sheets",
]


def get_file_ext(filename: str) -> str:
    return (filename or "").lower().rsplit(".", 1)[-1]


def detect_mime_type(filename: str) -> str:
    ext = get_file_ext(filename)
    return MIME_MAP.get(ext, "image/jpeg")


def get_file_category(filename: str) -> str:
    ext = get_file_ext(filename)
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in EXCEL_EXTENSIONS:
        return "excel"
    return "image"


def excel_to_text(file_bytes: bytes) -> str:
    """Convert Excel file to plain text for extraction."""
    try:
        import openpyxl
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        lines = []
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            lines.append(f"=== Sheet: {sheet_name} ===")
            for row in sheet.iter_rows(values_only=True):
                # Filter out completely empty rows
                row_values = [str(cell) if cell is not None else "" for cell in row]
                if any(v.strip() for v in row_values):
                    lines.append("\t".join(row_values))
        return "\n".join(lines)
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {str(e)}")


def safe_json_parse(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in model response")

    cleaned = text[start:end]
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)


def ensure_schema(data: dict) -> dict:
    for key in REQUIRED_KEYS:
        if key not in data or not isinstance(data[key], list):
            data[key] = []
    return data


# Matches "FLAT ROOM RATE 700/-", "flat room rate: 185", "FLAT ROOM RATE  90" etc.
FLAT_RATE_PATTERN = re.compile(
    r"flat\s*room\s*rate\s*[:\-]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def rescue_flat_room_rates(data: dict) -> dict:
    """
    Safety net: if the model left flat_room_rate null but the words
    'FLAT ROOM RATE <number>' are sitting inside any string field of a
    hotel row, pull that number into flat_room_rate. Never overwrites a
    value the model already extracted correctly. Only touches hotels.
    """
    hotels = data.get("hotels")
    if not isinstance(hotels, list):
        return data

    for hotel in hotels:
        if not isinstance(hotel, dict):
            continue

        existing = hotel.get("flat_room_rate")
        if isinstance(existing, (int, float)) and existing > 0:
            continue  # already correct, leave it

        found = None
        for value in hotel.values():
            if isinstance(value, str):
                m = FLAT_RATE_PATTERN.search(value)
                if m:
                    try:
                        found = float(m.group(1))
                    except ValueError:
                        found = None
                    if found is not None:
                        break

        if found is not None:
            hotel["flat_room_rate"] = int(found) if found.is_integer() else found
            # A genuine flat-room row should not also carry per-type rates
            # unless the model set them; leave any real numbers the model found.
            logger.info(
                "rescue_flat_room_rates: recovered flat_room_rate=%s for hotel '%s'",
                hotel["flat_room_rate"],
                hotel.get("name", "?"),
            )

    return data


PER_TYPE_COLUMNS = ["sharing", "quint", "quad", "triple", "double"]


# ── Single hotel rate sheets ─────────────────────────────────
#
# The model reads the grid; this code is what makes it safe to write to a
# database. Anything it cannot vouch for is dropped rather than guessed, because
# a wrong rate on a quote costs an agency real money.

ROOM_KEYS = {
    "sharing", "quint", "quad", "triple", "double", "flat_room_rate",
    "junior_suite_1", "junior_suite_2", "senior_suite_2", "senior_suite_3",
    "apartment", "executive_suite", "diplomatic_suite", "premium_suite",
    "royal_suite",
}

# What the sheets actually print, mapped onto the keys the app uses.
ROOM_ALIASES = {
    "twin": "double", "twindouble": "double", "twin/double": "double",
    "allexecutivetwindouble": "double", "executivetwindouble": "double",
    "singledouble": "double",
    "juniorsuite1": "junior_suite_1", "juniorsuiteone": "junior_suite_1",
    "juniorsuite2": "junior_suite_2", "juniorsuitetwo": "junior_suite_2",
    "seniorsuite2": "senior_suite_2", "senoirsuite2": "senior_suite_2",
    "seniorsuite3": "senior_suite_3", "senoirsuite3": "senior_suite_3",
    "executivesuite": "executive_suite", "businesssuite": "executive_suite",
    "executivebusinesssuite": "executive_suite", "executivebusinesssuites": "executive_suite",
    "diplomaticsuite": "diplomatic_suite", "diplomaticsuites": "diplomatic_suite",
    "premiumsuite": "premium_suite", "premiumsuites": "premium_suite",
    "royalsuite": "royal_suite", "royalsuites": "royal_suite",
    "apartments": "apartment",
    "flatroomrate": "flat_room_rate", "flatrate": "flat_room_rate", "roomrate": "flat_room_rate",
}

FLOOR_KEYS = {"regular", "premium", "executive"}

EXTRA_KEYS = {"breakfast", "lunch", "dinner", "half_board", "full_board", "parking", "other"}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
LOOSE_DATE = re.compile(r"^\s*(\d{1,2})[\s\-/]*([A-Za-z]{3,4})\.?[\s\-/]*(\d{2,4})?\s*$")


def _room_key(value):
    """Map a column header onto one of the app's room types, or give up."""
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if raw in ROOM_KEYS:
        return raw
    squashed = re.sub(r"[^a-z0-9]", "", raw)
    if squashed in ROOM_ALIASES:
        return ROOM_ALIASES[squashed]
    underscored = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if underscored in ROOM_KEYS:
        return underscored
    return None


def _iso_date(value, fallback_year):
    """A date we would be willing to price a stay on, or nothing."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    m = ISO_DATE.match(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2000 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return "%04d-%02d-%02d" % (y, mo, d)
        return None
    m = LOOSE_DATE.match(text)
    if not m:
        return None
    day = int(m.group(1))
    month = MONTHS.get(m.group(2).lower()[:4]) or MONTHS.get(m.group(2).lower()[:3])
    if not month or not (1 <= day <= 31):
        return None
    year_part = m.group(3)
    if year_part:
        year = int(year_part)
        if year < 100:
            year += 2000
    elif fallback_year:
        year = fallback_year
    else:
        return None
    return "%04d-%02d-%02d" % (year, month, day)


def _year_from_hint(hint):
    if not isinstance(hint, str):
        return None
    m = re.search(r"(20\d{2})", hint)
    return int(m.group(1)) if m else None


def _normalize_rate_sheets(data: dict) -> dict:
    """
    Take the model's grid and keep only what can be written without guessing.

    Columns that do not map to a known room type are dropped together with their
    whole column of prices, so a stray column can never shift a row of rates one
    place to the left. A period without two usable dates is dropped, because a
    seasonal rate with no season would silently price every night of the year.
    """
    sheets = data.get("rate_sheets")
    if not isinstance(sheets, list):
        data["rate_sheets"] = []
        return data

    clean_sheets = []
    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue
        hotel_name = (sheet.get("hotel") or sheet.get("name") or "").strip()
        if not hotel_name:
            continue
        fallback_year = _year_from_hint(sheet.get("year_hint"))

        clean_floors = []
        for floor in sheet.get("floors") or []:
            if not isinstance(floor, dict):
                continue
            rooms_in = floor.get("rooms")
            if not isinstance(rooms_in, list) or not rooms_in:
                continue

            labels_in = floor.get("room_labels") if isinstance(floor.get("room_labels"), list) else []
            details_in = floor.get("room_details") if isinstance(floor.get("room_details"), list) else []
            pax_in = floor.get("pax") if isinstance(floor.get("pax"), list) else []

            # Which columns survive, and where they sat in the original row.
            keep = []
            rooms, labels, details, pax = [], [], [], []
            seen = set()
            for i, raw in enumerate(rooms_in):
                key = _room_key(raw)
                if not key or key in seen:
                    continue
                seen.add(key)
                keep.append(i)
                rooms.append(key)
                labels.append(str(labels_in[i]).strip() if i < len(labels_in) and labels_in[i] else "")
                detail = str(details_in[i]).strip() if i < len(details_in) and details_in[i] else ""
                details.append(detail)
                cap = _num(pax_in[i]) if i < len(pax_in) else None
                pax.append(int(cap) if cap and 1 <= cap <= 20 else None)
            if not rooms:
                continue

            periods = []
            for period in floor.get("periods") or []:
                if not isinstance(period, dict):
                    continue
                rates_in = period.get("rates")
                if not isinstance(rates_in, list):
                    continue
                start = _iso_date(period.get("from"), fallback_year)
                end = _iso_date(period.get("to"), fallback_year)
                # A sheet with no dates at all is a standing price list, which is
                # fine. A half dated period is a misread, which is not.
                if (period.get("from") or period.get("to")) and not (start and end):
                    continue
                # A period that ends before it starts crossed the new year.
                if start and end and end < start:
                    try:
                        end = str(int(end[:4]) + 1) + end[4:]
                    except ValueError:
                        continue
                rates = []
                for i in keep:
                    value = _num(rates_in[i]) if i < len(rates_in) else None
                    rates.append(value if value and value > 0 else None)
                if not any(r is not None for r in rates):
                    continue
                periods.append({"from": start, "to": end, "rates": rates})
            if not periods:
                continue

            floor_key = (floor.get("floor") or "").strip().lower() if isinstance(floor.get("floor"), str) else ""
            bed = _num(floor.get("extra_bed"))
            clean_floors.append({
                "floor": floor_key if floor_key in FLOOR_KEYS else None,
                "covers": str(floor.get("covers") or "").strip(),
                "extra_bed": bed if bed and bed > 0 else None,
                "rooms": rooms,
                "room_labels": labels,
                "room_details": details,
                "pax": pax,
                "periods": periods,
            })
        if not clean_floors:
            continue

        clean_extras = []
        for extra in sheet.get("extras") or []:
            if not isinstance(extra, dict):
                continue
            kind = (extra.get("type") or "").strip().lower().replace(" ", "_")
            if kind not in EXTRA_KEYS:
                continue
            rate = _num(extra.get("rate"))
            if not rate or rate <= 0:
                continue
            floor_key = (extra.get("floor") or "").strip().lower() if isinstance(extra.get("floor"), str) else ""
            clean_extras.append({
                "type": kind,
                "floor": floor_key if floor_key in FLOOR_KEYS else None,
                "rate": rate,
            })

        clean_sheets.append({
            "hotel": hotel_name,
            "city": (sheet.get("city") or "").strip() or None,
            "distance": (sheet.get("distance") or None),
            "floors": clean_floors,
            "extras": clean_extras,
        })

    data["rate_sheets"] = clean_sheets
    return data


def normalize_rate_sheets(data: dict) -> dict:
    """
    The rate card is the newest thing this service does and the only one the
    older screens do not need. A sheet shaped in a way nobody anticipated must
    therefore cost an agency its rate card, never its whole upload: on any
    failure the key comes back empty and everything else is returned untouched.
    """
    try:
        return _normalize_rate_sheets(data)
    except Exception as e:
        logger.warning(f"Rate sheet normalisation failed, returning the rest: {e}")
        data["rate_sheets"] = []
        return data


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def dedupe_flat_into_columns(data: dict) -> dict:
    """
    Safety net for the opposite failure: the model reads a flat-room row
    correctly into flat_room_rate but ALSO copies that same number into a
    per-type column (almost always 'double', because the merged flat cell's
    right edge sits under the Double column). If a per-type column equals the
    flat_room_rate exactly, clear it to null. Only clears an exact duplicate,
    never a genuinely different per-type price.
    """
    hotels = data.get("hotels")
    if not isinstance(hotels, list):
        return data

    for hotel in hotels:
        if not isinstance(hotel, dict):
            continue

        flat = _num(hotel.get("flat_room_rate"))
        if flat is None or flat <= 0:
            continue

        for col in PER_TYPE_COLUMNS:
            col_val = _num(hotel.get(col))
            if col_val is not None and col_val == flat:
                hotel[col] = None
                logger.info(
                    "dedupe_flat_into_columns: cleared %s=%s (duplicate of flat_room_rate) for hotel '%s'",
                    col,
                    int(flat) if flat.is_integer() else flat,
                    hotel.get("name", "?"),
                )

    return data


def _row_rate_value(hotel: dict):
    """The single meaningful rate on a hotel row: flat first, else any per-type."""
    flat = _num(hotel.get("flat_room_rate"))
    if flat is not None and flat > 0:
        return flat
    for col in PER_TYPE_COLUMNS:
        v = _num(hotel.get(col))
        if v is not None and v > 0:
            return v
    return None


def flag_repeated_rates(data: dict, run_length: int = 3) -> dict:
    """
    Copy-down errors show up as the same rate repeating on consecutive rows.
    Some repeats are genuine (four hotels really at 250), so we don't change the
    numbers — we only lower confidence so the human checks them. Any run of
    `run_length` or more consecutive hotels (same city) with the identical rate
    gets confidence='low'. Never raises confidence the model already set to low.
    """
    hotels = data.get("hotels")
    if not isinstance(hotels, list) or not hotels:
        return data

    # Walk city by city, in order.
    i = 0
    n = len(hotels)
    while i < n:
        h = hotels[i]
        if not isinstance(h, dict):
            i += 1
            continue

        city = (h.get("city") or "").lower()
        val = _row_rate_value(h)

        j = i + 1
        while j < n:
            nxt = hotels[j]
            if not isinstance(nxt, dict):
                break
            if (nxt.get("city") or "").lower() != city:
                break
            if _row_rate_value(nxt) != val or val is None:
                break
            j += 1

        run = j - i
        if val is not None and run >= run_length:
            for k in range(i, j):
                if isinstance(hotels[k], dict):
                    hotels[k]["confidence"] = "low"
            logger.info(
                "flag_repeated_rates: %d consecutive rows share rate %s in %s -> flagged low",
                run,
                int(val) if float(val).is_integer() else val,
                city or "?",
            )

        i = j if j > i + 1 else i + 1

    return data


def build_message_content(
    file_bytes: bytes,
    filename: str,
    instructions: str,
) -> list:
    category = get_file_category(filename)

    instruction_block = ""
    if instructions.strip():
        instruction_block = f"\n\nCUSTOM INSTRUCTIONS (follow these strictly during extraction):\n{instructions.strip()}"

    extraction_prompt = f"Extract all structured pricing data from this rate sheet. Follow all extraction rules strictly.{instruction_block}"

    # ── Plain text ────────────────────────────────────────────────────────────
    if category == "text":
        try:
            text_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = file_bytes.decode("latin-1")

        return [
            {
                "type": "text",
                "text": f"{extraction_prompt}\n\nRATE SHEET CONTENT:\n{text_content}",
            }
        ]

    # ── Excel ─────────────────────────────────────────────────────────────────
    if category == "excel":
        text_content = excel_to_text(file_bytes)
        return [
            {
                "type": "text",
                "text": f"{extraction_prompt}\n\nRATE SHEET CONTENT (from Excel):\n{text_content}",
            }
        ]

    # ── Image or PDF ──────────────────────────────────────────────────────────
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")
    mime_type = detect_mime_type(filename)

    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            },
        },
        {
            "type": "text",
            "text": extraction_prompt,
        },
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Umrah Rate Sheet Extractor",
        "model": MODEL,
        "version": "1.0.0",
        "supported_formats": ["jpg", "jpeg", "png", "webp", "gif", "pdf", "txt", "csv", "xlsx", "xls"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/extract")
async def extract_data(
    file: UploadFile,
    instructions: str = Form(default=""),
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is not set.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload.jpg"
    category = get_file_category(filename)

    logger.info(f"Processing: {filename} | type: {category} | size: {len(file_bytes)} bytes")
    if instructions.strip():
        logger.info(f"Custom instructions provided: {instructions.strip()[:200]}")

    try:
        user_content = build_message_content(file_bytes, filename, instructions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = {
        "model": MODEL,
        "temperature": 0,
        # A single hotel grid is three floors of six dated rows, which runs far
        # past the old ceiling and used to come back truncated.
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://umrah-extractor.onrender.com",
        "X-Title": "Umrah Rate Sheet Extractor",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenRouter request timed out. Try again.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error: {str(e)}")

    if response.status_code != 200:
        logger.error(f"OpenRouter error {response.status_code}: {response.text[:500]}")
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {response.status_code}: {response.text[:300]}",
        )

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected response from OpenRouter: {str(e)}")

    logger.info(f"Model response (first 300 chars): {content[:300]}")

    try:
        parsed = safe_json_parse(content)
    except Exception as e:
        logger.error(f"JSON parse failed: {content[:1000]}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model returned invalid JSON. Try again.",
                "details": str(e),
                "raw_response": content[:1000],
            },
        )

    parsed = ensure_schema(parsed)
    parsed = rescue_flat_room_rates(parsed)
    parsed = dedupe_flat_into_columns(parsed)
    parsed = flag_repeated_rates(parsed)
    parsed = normalize_rate_sheets(parsed)
    logger.info("Extraction successful.")
    return parsed

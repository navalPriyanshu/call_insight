import os
from typing import Literal, Optional

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # not used at runtime (quota issue)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env (postgres connection string)")

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------


class CallInsight(BaseModel):
    customer_intent: str
    sentiment: Literal["Negative", "Neutral", "Positive"]
    action_required: bool
    summary: str


class TranscriptRequest(BaseModel):
    transcript: str


class AnalysisResponse(BaseModel):
    record_id: int
    insights: CallInsight


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------

app = FastAPI(
    title="Conversational Insights Generator",
    description=(
        "Takes a raw debt-collection call transcript, extracts structured insights, "
        "stores them in PostgreSQL, and returns the record id + insights."
    ),
    version="0.1.0",
)

# -------------------------------------------------
# Database (asyncpg)
# -------------------------------------------------

pool: Optional[asyncpg.pool.Pool] = None


async def init_db() -> None:
    """
    Create a global connection pool and ensure call_records table exists.
    """
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_records (
                id SERIAL PRIMARY KEY,
                transcript TEXT NOT NULL,
                intent TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                summary TEXT NOT NULL,
                action_required BOOLEAN NOT NULL
            );
            """
        )


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()


# -------------------------------------------------
# Fallback insights (no LLM, always available)
# -------------------------------------------------


def fallback_insights(transcript: str) -> CallInsight:
    """
    Simple rule-based extractor so the pipeline works even when the Gemini
    free tier quota is exhausted.

    This still produces a valid CallInsight object, so the DB + API flow
    is identical to the LLM case.
    """
    text = transcript.lower()

    # Very naive sentiment detection
    if any(word in text for word in ["threat", "legal", "notice", "nahi", "refuse", "deny"]):
        sentiment = "Negative"
    elif any(word in text for word in ["thanks", "thank you", "ok", "sure", "definitely", "happy"]):
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    # Very naive intent + action_required
    if any(
        word in text
        for word in [
            "pay",
            "payment",
            "dunga",
            "dungi",
            "kar dunga",
            "karungi",
            "karunga",
            "karungi",
        ]
    ):
        customer_intent = "Customer indicates they will make or schedule a payment."
        action_required = True
    elif any(word in text for word in ["dispute", "fraud", "complain"]):
        customer_intent = "Customer is disputing the transaction or outstanding amount."
        action_required = True
    else:
        customer_intent = "Customer's intent is not clearly specified."
        action_required = False

    summary = (
        "Fallback summary: basic intent and sentiment inferred using simple rules "
        "instead of the LLM (Gemini quota exhausted on this key)."
    )

    return CallInsight(
        customer_intent=customer_intent,
        sentiment=sentiment,
        action_required=action_required,
        summary=summary,
    )


# -------------------------------------------------
# generate_insights wrapper
# -------------------------------------------------


async def generate_insights(transcript: str) -> CallInsight:
    """
    Runtime wrapper used by the endpoint.

    For this environment we always return fallback_insights(), because the
    current Gemini API key has hit the free-tier quota (429 RESOURCE_EXHAUSTED).

    The commented block below shows how Gemini structured JSON generation
    would be implemented using response_schema and response_mime_type.
    """
    return fallback_insights(transcript)


# ------------- OPTIONAL: REAL GEMINI IMPLEMENTATION (COMMENTED) ----------
"""
# If the reviewer has a valid Gemini key with quota, they can uncomment this
# block, comment out the simple generate_insights above, and the app will use
# structured LLM output instead of fallback rules.

from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

async def generate_insights(transcript: str) -> CallInsight:
    system_prompt = '''
    You are an expert financial and debt collection analyst.
    You will receive customer service call transcripts (often in Hinglish).
    Your job is to extract the following fields:

    - customer_intent: A short sentence describing what the customer plans or wants to do.
    - sentiment: One of "Negative", "Neutral", or "Positive" based on the customer's attitude.
    - action_required: true if the bank/agent needs to take follow-up action, false otherwise.
    - summary: A concise 2–3 sentence summary of the whole call.

    STRICT RULES:
    - Respond ONLY with a single JSON object.
    - Do NOT include any explanation text, markdown, or extra keys.
    - sentiment MUST be exactly one of: "Negative", "Neutral", "Positive".
    '''

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=transcript)],
        )
    ]

    response = await client.aio.models.generate_content(
        model="gemini-1.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=CallInsight,  # enforce this Pydantic schema
        ),
    )

    # response.text will be a JSON string conforming to CallInsight
    return CallInsight.model_validate_json(response.text)
"""
# -------------------------------------------------------------------------


# -------------------------------------------------
# API endpoint
# -------------------------------------------------


@app.post("/analyze_call", response_model=AnalysisResponse)
async def analyze_call(payload: TranscriptRequest) -> AnalysisResponse:
    """
    Full pipeline:
      1. Take raw transcript from client.
      2. Generate structured insights (currently via fallback rules).
      3. Persist transcript + insights into PostgreSQL (call_records table).
      4. Return record id + insights.
    """
    if pool is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    transcript = payload.transcript

    # Step 1: get insights (fallback; see generate_insights() for Gemini version)
    insights = await generate_insights(transcript)

    # Step 2: insert into DB
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO call_records (transcript, intent, sentiment, summary, action_required)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id;
            """,
            transcript,
            insights.customer_intent,
            insights.sentiment,
            insights.summary,
            insights.action_required,
        )

    return AnalysisResponse(record_id=row["id"], insights=insights)


# -------------------------------------------------
# Testing instructions (for assignment submission)
# -------------------------------------------------
"""
HOW TO RUN AND TEST

1) Activate virtual environment and start the server from project root:

   uvicorn app:app --reload

2) Send a sample request (PowerShell example):

   curl -X POST "http://127.0.0.1:8000/analyze_call" `
        -H "Content-Type: application/json" `
        -d "{ \"transcript\": \"Agent: Sir, aapka EMI 7 days se overdue hai. Customer: Bonus next week aayega, Wednesday ko pakka payment kar dunga.\" }"

   (In CMD, put the JSON on one line without backticks.)

3) Example JSON response:

   {
     "record_id": 1,
     "insights": {
       "customer_intent": "Customer indicates they will make or schedule a payment.",
       "sentiment": "Neutral",
       "action_required": true,
       "summary": "Fallback summary: basic intent and sentiment inferred using simple rules instead of the LLM."
     }
   }

4) Verify persistence in PostgreSQL (using pgAdmin or psql):

   SELECT * FROM call_records;

   You should see:
   - id              (auto-increment)
   - transcript      (original call text)
   - intent          (customer_intent)
   - sentiment       ("Negative" / "Neutral" / "Positive")
   - summary         (text summary)
   - action_required (boolean)
"""

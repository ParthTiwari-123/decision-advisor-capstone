"""
Market Analysis Agent
Uses Google Search grounding to find real-time market data,
then returns JSON: {opportunities, threats, timing}
"""

import os
import json
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.0-flash"

SYSTEM_INSTRUCTION = """You are a market analysis expert. Given a decision the user is considering,
research current market conditions, industry trends, and economic factors using the search tool.

You MUST respond with ONLY valid JSON — no markdown, no code fences, no extra text.
Use this exact schema:
{
  "opportunities": ["opportunity 1", "opportunity 2", ...],
  "threats": ["threat 1", "threat 2", ...],
  "timing": "Your assessment of whether NOW is a good time and why"
}

Keep each array to 3-5 items. Be specific and data-driven."""


async def run_market_agent(decision: str, timeline: str) -> dict:
    """Analyse market conditions relevant to the user's decision."""
    prompt = (
        f"Analyse the market landscape for this decision: '{decision}'. "
        f"Timeline: {timeline}. "
        "Search for current trends, competitor activity, and economic indicators. "
        "Return your analysis as JSON only."
    )

    google_search_tool = Tool(google_search=GoogleSearch())

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=500,
            tools=[google_search_tool],
        ),
    )

    try:
        text = response.text.strip()
        # Strip markdown code fences if the model wraps them
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {
            "opportunities": ["Unable to parse market data"],
            "threats": ["Analysis encountered an error"],
            "timing": response.text if response.text else "No response from model",
        }

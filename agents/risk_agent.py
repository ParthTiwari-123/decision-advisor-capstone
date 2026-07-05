"""
Risk Assessment Agent
Evaluates risk factors for a decision and returns structured JSON:
{score_1_to_10, best_case, worst_case, recommendation}
"""

import os
import json
from google import genai
from google.genai.types import GenerateContentConfig

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.0-flash"

SYSTEM_INSTRUCTION = """You are a risk assessment specialist. Evaluate the risk profile of the
user's decision based on their stated risk tolerance, priority, and timeline.

You MUST respond with ONLY valid JSON — no markdown, no code fences, no extra text.
Use this exact schema:
{
  "score_1_to_10": <integer 1-10 where 10 is highest risk>,
  "best_case": "Description of the best realistic outcome",
  "worst_case": "Description of the worst realistic outcome",
  "recommendation": "Your specific, actionable recommendation"
}

Be honest and specific. Reference the user's risk tolerance in your recommendation."""


async def run_risk_agent(
    decision: str, risk_tolerance: str, priority: str, timeline: str
) -> dict:
    """Assess risk for the given decision context."""
    prompt = (
        f"Evaluate the risk of this decision: '{decision}'.\n"
        f"Risk tolerance: {risk_tolerance}\n"
        f"Top priority: {priority}\n"
        f"Timeline: {timeline}\n\n"
        "Provide a thorough risk assessment as JSON only."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=500,
        ),
    )

    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {
            "score_1_to_10": 5,
            "best_case": "Unable to parse risk data",
            "worst_case": "Analysis encountered an error",
            "recommendation": response.text if response.text else "No response from model",
        }

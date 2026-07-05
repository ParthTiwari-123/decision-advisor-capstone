"""
Psychology / Bias Detection Agent
Analyses past decisions for cognitive biases and returns:
{biases[], alignment_score, gut_check_question}
"""

import os
import json
from google import genai
from google.genai.types import GenerateContentConfig

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.0-flash"

SYSTEM_INSTRUCTION = """You are a behavioral psychology expert specialising in cognitive biases
and decision-making patterns. Given the user's current decision and their past decisions,
detect cognitive biases that may be influencing them.

You MUST respond with ONLY valid JSON — no markdown, no code fences, no extra text.
Use this exact schema:
{
  "biases": [
    {"name": "Bias Name", "description": "How this bias is showing up in their pattern"}
  ],
  "alignment_score": <integer 1-10 where 10 means fully aligned with stated priorities>,
  "gut_check_question": "A single powerful question to help the user think more clearly"
}

Include 2-4 biases. The gut-check question should be thought-provoking and specific to their situation."""


async def run_psychology_agent(
    decision: str, priority: str, past_decisions: str
) -> dict:
    """Detect cognitive biases from past decision patterns."""
    prompt = (
        f"Current decision being considered: '{decision}'\n"
        f"Stated top priority: {priority}\n"
        f"Past decisions and context: {past_decisions}\n\n"
        "Analyse for cognitive biases, assess alignment with stated priorities, "
        "and provide a gut-check question. Return JSON only."
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
            "biases": [{"name": "Parse Error", "description": "Could not parse agent response"}],
            "alignment_score": 5,
            "gut_check_question": response.text if response.text else "No response from model",
        }

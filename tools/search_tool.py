"""
Google Search grounding tool for the Market Agent.
Uses the google.genai built-in google_search tool.
"""

from google.genai.types import Tool, GoogleSearch

# The built-in Google Search grounding tool for ADK agents
google_search_tool = Tool(google_search=GoogleSearch())

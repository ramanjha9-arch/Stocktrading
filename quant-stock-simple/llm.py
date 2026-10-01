"""
Gemini is used ONLY to write the reasoning paragraph for a stock the scoring
engine already shortlisted — it receives the computed numbers as context and
is explicitly told not to invent any figure not given to it. If no API key
is configured, reasoning is simply omitted (never faked).
"""
import os
import google.generativeai as genai

_configured = False


def _client():
    global _configured
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if not _configured:
        genai.configure(api_key=api_key)
        _configured = True
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return genai.GenerativeModel(model_name)


PROMPT_TEMPLATE = """You are a research assistant explaining a quantitative stock score to a retail \
investor. You are NOT choosing the stock — it was already shortlisted by a rule-based scoring \
system. Use ONLY the numbers given below. Do not invent any price, percentage, or fact not listed. \
Do not state or imply a guaranteed return. If the data is thin or mixed, say so plainly.

Stock: {symbol}
Horizon: {horizon}
Composite score (0-100): {composite}
Technical score: {technical} — details: {tech_detail}
Fundamental score: {fundamental} — details: {fund_detail}
Sentiment score: {sentiment} — details: {sent_detail}
Recent relevant headlines: {headlines}

Write 3-5 sentences: (1) what the technical picture shows, (2) what the fundamentals/news \
add or subtract, (3) one honest caveat or risk. Plain language, no bullet points, no disclaimers \
about being an AI."""


def generate_reasoning(symbol, horizon, composite, technical, tech_detail,
                        fundamental, fund_detail, sentiment, sent_detail, headlines) -> str | None:
    model = _client()
    if model is None:
        return None
    prompt = PROMPT_TEMPLATE.format(
        symbol=symbol, horizon=horizon, composite=composite,
        technical=technical if technical is not None else "unavailable",
        tech_detail=tech_detail, fundamental=fundamental if fundamental is not None else "unavailable",
        fund_detail=fund_detail, sentiment=sentiment if sentiment is not None else "unavailable",
        sent_detail=sent_detail, headlines="; ".join(headlines) if headlines else "none in the last 7 days",
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[LLM reasoning unavailable: {e}]"

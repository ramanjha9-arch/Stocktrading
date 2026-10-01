"""
Condensed rule-based news classifier. Direction comes from the matched EVENT
TYPE, not generic sentiment words — "announces new plant" stays neutral
unless a disclosed amount can be sized against market cap.
"""
import re

RULES = [
    ("fraud_allegation", -1, r"\bfraud|forg(ed|ery)|siphon|embezzle|money laundering|cbi raid", 0.8),
    ("regulatory_action", -1, r"\bsebi (order|bans?|penal|probe)|penalty|barred|debarred|licen[cs]e (cancel|suspend)", 0.8),
    ("litigation", -1, r"\blawsuit|litigation|sued\b|nclt|insolvency", 0.6),
    ("credit_downgrade", -1, r"downgrad(e|es|ed).*(rating|crisil|icra|care|moody|fitch)|rating (cut|lowered)", 0.85),
    ("debt_stress", -1, r"\bdefault(s|ed)?\b|debt (stress|restructur)|missed (payment|interest)", 0.85),
    ("order_cancellation", -1, r"order (cancel|cancelled|terminated)|loses? (order|contract)", 0.75),
    ("earnings_miss", -1, r"miss(es|ed)? (estimates|expectations)|profit (falls?|slumps?|plunges?|declines?)|net loss|reports? loss", 0.6),
    ("negative_guidance", -1, r"(cuts?|lowers?|slashes) (guidance|outlook|forecast)|weak (guidance|outlook)", 0.65),
    ("earnings_beat", 1, r"(beats?|tops?|surpass(es)?) (estimates|expectations)", 0.75),
    ("earnings_growth", 1, r"(net )?profit (jumps?|surges?|rises?|soars?|climbs?)|record (profit|revenue)", 0.55),
    ("regulatory_approval", 1, r"(usfda|fda|dcgi|cci|nclt) (approv|clearance|clears?)|receives? approval", 0.8),
    ("debt_reduction", 1, r"debt (reduction|repay|free)|repays? (debt|loan)|deleverag", 0.7),
    ("positive_guidance", 1, r"(raises?|lifts?|hikes?|upgrad(es|ed)?) (guidance|outlook|forecast|target)", 0.65),
    ("credit_upgrade", 1, r"upgrad(e|es|ed).*(rating|crisil|icra|care|moody|fitch)", 0.8),
    # size-dependent: direction only counts if a disclosed amount is found
    ("major_order", 1, r"(bags?|wins?|secures?|lands?) .*(order|contract|deal)|order (win|inflow)", 0.6),
    ("capacity_expansion", 1, r"(sets? up|to set up|commission(s|ed)?|expands?|expansion|capex|new plant|new facility)", 0.5),
    ("acquisition", 1, r"acquir(e|es|ed|ing)|acquisition|takeover|merger", 0.5),
]
_COMPILED = [(t, d, re.compile(p, re.I), c) for t, d, p, c in RULES]
_DENIAL = re.compile(r"\b(denies|denied|refutes|clarifies|dismisses|baseless|rumou?rs?)\b", re.I)
_SIZE_DEPENDENT = {"major_order", "capacity_expansion", "acquisition"}
_AMOUNT = re.compile(r"(?:₹|rs\.?|inr)\s?([\d,]+(?:\.\d+)?)\s?(crore|cr|lakh)?", re.I)
MATERIALITY_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.25, "unknown": 0.3}


def _parse_amount_cr(text: str) -> float | None:
    best = None
    for m in _AMOUNT.finditer(text):
        try:
            num = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        cr = num if unit in ("crore", "cr") else num / 100 if unit == "lakh" else None
        if cr is not None:
            best = cr if best is None else max(best, cr)
    return best


def classify_headline(headline: str, market_cap: float | None = None) -> dict | None:
    for etype, direction, pat, conf in _COMPILED:
        if not pat.search(headline):
            continue
        amount = _parse_amount_cr(headline)
        if amount is not None and market_cap:
            ratio = (amount * 1e7) / market_cap
            materiality = "high" if ratio >= 0.05 else "medium" if ratio >= 0.01 else "low"
        elif amount is not None:
            materiality = "unknown"
        else:
            materiality = "unknown"

        if etype in _SIZE_DEPENDENT and materiality == "unknown":
            direction, conf = 0, min(conf, 0.3)
        if direction < 0 and _DENIAL.search(headline):
            conf *= 0.5

        score = round(direction * MATERIALITY_WEIGHT[materiality] * conf, 4)
        return {"event_type": etype, "direction": direction, "materiality": materiality,
                "confidence": round(conf, 3), "event_score": score}
    return None

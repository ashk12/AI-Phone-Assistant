import re
ADVERSARIAL_PATTERNS = [
    r"reveal.*prompt",
    r"show.*api.?key",
    r"ignore.*rules",
    r"internal.*logic",
    r"jailbreak",
    r"(bypass|disable).*safety",
    r"trash|insult|hate|defame|attack|bias",
]

def is_unsafe_query(query: str) -> bool:
    """Detect obvious unsafe or adversarial patterns"""
    q = query.lower().strip()
    return any(re.search(p, q) for p in ADVERSARIAL_PATTERNS)

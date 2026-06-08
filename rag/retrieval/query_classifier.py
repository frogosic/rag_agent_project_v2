import re


_EXACT_LOOKUP_PATTERN = re.compile(
    r"^([A-Z0-9_./:-]+|[A-Z]+\s+/[a-z0-9_./:-]+|[a-z0-9_.:-]+)$"
)


def is_exact_lookup_query(query: str) -> bool:
    stripped = query.strip()

    if not stripped:
        return False

    return bool(_EXACT_LOOKUP_PATTERN.fullmatch(stripped))

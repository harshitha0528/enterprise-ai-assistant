import re
from collections import Counter
from pathlib import Path
from uuid import uuid4

LOG_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) \[([\w\-]+)\] (.*)$"
)


def parse_log(text: str) -> list[dict]:
    """Turn raw log text into a list of structured entries."""
    entries = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = LOG_PATTERN.match(line.strip())
        if match:
            timestamp, level, service, message = match.groups()
            entries.append({
                "line": line_number,
                "timestamp": timestamp,
                "level": level.upper(),
                "service": service,
                "message": message,
            })
    return entries


def normalize(message: str) -> str:
    """Replace numbers so 'order 1001' and 'order 1002' count as the same message."""
    return re.sub(r"\d+", "#", message)


def summarize_log(entries: list[dict]) -> dict:
    errors = [e for e in entries if e["level"] == "ERROR"]
    warnings = [e for e in entries if e["level"] in ("WARN", "WARNING")]

    repeated = Counter((e["service"], normalize(e["message"])) for e in errors + warnings)

    return {
        "total_entries": len(entries),
        "by_level": dict(Counter(e["level"] for e in entries)),
        "errors_by_service": dict(Counter(e["service"] for e in errors)),
        "first_error": errors[0]["timestamp"] if errors else None,
        "last_error": errors[-1]["timestamp"] if errors else None,
        "repeated_problems": [
            {"service": service, "message": message, "count": count}
            for (service, message), count in repeated.most_common()
            if count > 1
        ],
    }

LOG_DIR = Path(__file__).resolve().parents[1] / "storage" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def save_log(content: bytes) -> str:
    log_id = uuid4().hex
    (LOG_DIR / f"{log_id}.log").write_bytes(content)
    return log_id


def load_entries(log_id: str) -> list[dict]:
    # Validate the ID so nobody can pass something like ../../secret to read other files
    if not LOG_ID_PATTERN.match(log_id):
        raise ValueError("Invalid log id")
    path = LOG_DIR / f"{log_id}.log"
    if not path.exists():
        raise FileNotFoundError("Log not found")
    return parse_log(path.read_text(encoding="utf-8", errors="replace"))


# ---- Investigation tools (the agent will call these later) ----

def search_logs(entries: list[dict], keyword: str, limit: int = 20) -> list[dict]:
    """Find log lines whose message or service contains the keyword."""
    kw = keyword.lower()
    matches = [e for e in entries if kw in e["message"].lower() or kw in e["service"].lower()]
    return matches[:limit]


def problems_between(entries: list[dict], start: str, end: str) -> list[dict]:
    """Errors and warnings inside a time window (format: 2026-10-28 09:14:00)."""
    return [
        e for e in entries
        if e["level"] in ("ERROR", "WARN", "WARNING") and start <= e["timestamp"] <= end
    ]


def problem_timeline(entries: list[dict]) -> list[dict]:
    """Each distinct problem with when it first appeared and how often. Shows what came first."""
    seen = {}
    for e in entries:
        if e["level"] not in ("ERROR", "WARN", "WARNING"):
            continue
        key = (e["service"], normalize(e["message"]))
        if key not in seen:
            seen[key] = {"service": e["service"], "message": key[1],
                         "first_seen": e["timestamp"], "count": 0}
        seen[key]["count"] += 1
    return sorted(seen.values(), key=lambda p: p["first_seen"])    
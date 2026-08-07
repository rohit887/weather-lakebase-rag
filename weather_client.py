"""Client for the National Weather Service API (api.weather.gov).

No API key is required, but every request MUST send a User-Agent header
containing a contact email or NWS returns 403. Edit CONTACT_EMAIL /
USER_AGENT below to a real address you control.
"""

import hashlib
from datetime import datetime, timezone

import requests

# --- Edit me: NWS blocks requests without a real contact email in the UA. -----
CONTACT_EMAIL = "rohit885@gmail.com"
USER_AGENT = f"weather-lakebase-rag/1.0 ({CONTACT_EMAIL})"
# -----------------------------------------------------------------------------

API_BASE = "https://api.weather.gov"
REQUEST_TIMEOUT = 15  # seconds

# Small hardcoded gazetteer. Keys are "City, ST" exactly as callers pass them.
CITY_COORDS = {
    "Chicago, IL": (41.8781, -87.6298),
    "Austin, TX": (30.2672, -97.7431),
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Denver, CO": (39.7392, -104.9903),
    "Miami, FL": (25.7617, -80.1918),
    "Seattle, WA": (47.6062, -122.3321),
    "New Orleans, LA": (29.9511, -90.0715),
}

# City -> (office, gridX, gridY). Cached in-process so we resolve /points once.
_GRID_CACHE = {}


def _headers():
    return {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}


def _get(url, params=None):
    """GET a NWS URL, returning parsed JSON or None on any error/timeout.

    Never raises to the caller: a flaky NWS response should not abort a sync.
    """
    try:
        resp = requests.get(
            url, params=params, headers=_headers(), timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def _state_of(location):
    """Extract the two-letter state code from a 'City, ST' string."""
    if "," not in location:
        return None
    return location.split(",")[-1].strip().upper()


def _resolve_grid(location):
    """Two-hop lookup: /points/{lat},{lon} -> (office, gridX, gridY).

    Cached in _GRID_CACHE so repeated syncs for the same city don't re-resolve.
    Returns None if the location is unknown or NWS can't be reached.
    """
    if location in _GRID_CACHE:
        return _GRID_CACHE[location]

    coords = CITY_COORDS.get(location)
    if coords is None:
        return None
    lat, lon = coords

    data = _get(f"{API_BASE}/points/{lat},{lon}")
    if not data:
        return None

    props = data.get("properties", {})
    office = props.get("gridId")
    grid_x = props.get("gridX")
    grid_y = props.get("gridY")
    if office is None or grid_x is None or grid_y is None:
        return None

    grid = (office, grid_x, grid_y)
    _GRID_CACHE[location] = grid
    return grid


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts):
    """Deterministic id from the given parts (used for forecast periods)."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _fetch_alerts(location):
    """Fetch active alerts for the location's state and normalize them."""
    state = _state_of(location)
    if not state:
        return []

    data = _get(f"{API_BASE}/alerts/active", params={"area": state})
    if not data:
        return []

    records = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            continue

        description = props.get("description") or ""
        instruction = props.get("instruction") or ""
        narrative = "\n\n".join(part for part in (description, instruction) if part)
        if not narrative:
            continue

        records.append(
            {
                "id": alert_id,
                "location": location,
                "source_type": "alert",
                "headline": props.get("event"),
                "narrative_text": narrative,
                "issued_at": props.get("sent") or props.get("effective"),
                "payload": props,
                "synced_at": _now_iso(),
            }
        )
    return records


def _fetch_forecast(location):
    """Fetch the multi-period forecast and normalize one record per period."""
    grid = _resolve_grid(location)
    if grid is None:
        return []
    office, grid_x, grid_y = grid

    data = _get(f"{API_BASE}/gridpoints/{office}/{grid_x},{grid_y}/forecast")
    if not data:
        return []

    records = []
    for period in data.get("properties", {}).get("periods", []):
        detailed = period.get("detailedForecast") or ""
        if not detailed:
            continue

        start_time = period.get("startTime")
        doc_id = _stable_id(location, start_time)

        records.append(
            {
                "id": doc_id,
                "location": location,
                "source_type": "forecast",
                "headline": period.get("name"),
                "narrative_text": detailed,
                "issued_at": start_time,
                "payload": period,
                "synced_at": _now_iso(),
            }
        )
    return records


def fetch_location_documents(location):
    """Fetch alerts + forecast for one 'City, ST' location.

    Returns a list of normalized records with the shape:
    id, location, source_type, headline, narrative_text, issued_at,
    payload, synced_at. Errors and timeouts yield fewer (or zero) records
    rather than raising.
    """
    return _fetch_alerts(location) + _fetch_forecast(location)

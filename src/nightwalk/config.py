"""Study area, scoring weights and routing parameters.

Every number here is a modelling choice, not a measured fact. They are
collected in one place so the sensitivity analysis can vary them and the
README can state them.
"""
from datetime import datetime

# ── Study area ──────────────────────────────────────────────────────────
# Kolkata core: Kolkata Municipal Corporation plus Salt Lake / New Town
# edge (south, west, north, east).
BBOX = (22.47, 88.30, 22.62, 88.43)
UTM_EPSG = 32645

# Hour of the evaluation. A Wednesday avoids weekend-only openings.
NIGHT_TIME = datetime(2026, 11, 4, 22, 0)
# Hours offered on the map. Midnight and 02:00 fall on the Thursday.
NIGHT_HOURS = [datetime(2026, 11, 4, 20, 0), datetime(2026, 11, 4, 22, 0),
               datetime(2026, 11, 5, 0, 0), datetime(2026, 11, 5, 2, 0)]

# ── Walkable network ────────────────────────────────────────────────────
WALKABLE = {
    "primary", "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "residential", "living_street", "unclassified",
    "service", "pedestrian", "footway", "path", "steps", "track",
    "trunk", "trunk_link", "corridor", "road",
}
EXCLUDE_FOOT = {"no", "private"}

# ── Feature extraction ──────────────────────────────────────────────────
POI_RADIUS_M = 50          # places that put "eyes on the street"
LAMP_RADIUS_M = 20         # street lamp counted for a segment
GUARDIAN_RADIUS_M = 150    # police, hospital emergency, 24 h fuel
ISOLATION_RADIUS_M = 30    # land-use buffer for isolation
LAMP_SPACING_FULL_M = 35   # one lamp per 35 m counts as fully lit

# Land uses that leave a street without frontage after dark
ISOLATING_LANDUSE = {
    ("leisure", "park"), ("leisure", "garden"), ("landuse", "cemetery"),
    ("amenity", "grave_yard"), ("landuse", "railway"), ("landuse", "industrial"),
    ("landuse", "brownfield"), ("landuse", "construction"),
    ("natural", "water"), ("natural", "wetland"), ("landuse", "military"),
}

# Probability that a place WITHOUT opening_hours is open at 22:00. Used only
# when the tag is missing; every use lowers that segment's confidence.
OPEN_PRIOR = {
    "restaurant": 0.6, "fast_food": 0.6, "cafe": 0.3, "bar": 0.8, "pub": 0.8,
    "fuel": 0.7, "pharmacy": 0.4, "hospital": 1.0, "clinic": 0.2,
    "convenience": 0.5, "supermarket": 0.3, "kiosk": 0.3, "hotel": 1.0,
    "cinema": 0.7, "police": 1.0, "bank": 0.0, "atm": 0.5,
    "place_of_worship": 0.2, "school": 0.0, "college": 0.0,
    "university": 0.0, "marketplace": 0.4,
}
OPEN_PRIOR_DEFAULT_SHOP = 0.25
OPEN_PRIOR_DEFAULT = 0.15
# The priors above are for 22:00. At other hours they are scaled by this
# factor (capped at 1), except for places that never close.
HOUR_FACTOR = {20: 1.6, 21: 1.3, 22: 1.0, 23: 0.6, 0: 0.3, 1: 0.2, 2: 0.15,
               3: 0.1, 4: 0.1, 5: 0.2}
ALWAYS_OPEN = {"hospital", "hotel", "police"}

# ── Scoring ─────────────────────────────────────────────────────────────
WEIGHTS = {"lighting": 0.40, "activity": 0.35, "context": 0.25}

# Road class -> prior for how busy and overlooked a street is at night.
ROAD_CONTEXT = {
    "primary": 0.9, "primary_link": 0.8, "trunk": 0.7, "trunk_link": 0.6,
    "secondary": 0.85, "secondary_link": 0.75, "tertiary": 0.75,
    "tertiary_link": 0.7, "residential": 0.55, "living_street": 0.55,
    "unclassified": 0.45, "road": 0.45, "pedestrian": 0.6, "service": 0.35,
    "footway": 0.4, "corridor": 0.4, "path": 0.2, "track": 0.15, "steps": 0.25,
}
# Road class -> prior probability of street lighting where lit=* is absent
LIT_PRIOR = {
    "primary": 0.9, "primary_link": 0.85, "trunk": 0.9, "trunk_link": 0.85,
    "secondary": 0.85, "secondary_link": 0.8, "tertiary": 0.7,
    "tertiary_link": 0.7, "residential": 0.5, "living_street": 0.5,
    "unclassified": 0.4, "road": 0.4, "pedestrian": 0.6, "service": 0.3,
    "footway": 0.35, "corridor": 0.5, "path": 0.1, "track": 0.05, "steps": 0.3,
}
ACTIVITY_HALF = 3.0        # open places within 50 m giving activity 0.5
# Where OSM maps no place within 50 m, absence is not evidence of an empty
# street (most small shops in Kolkata are unmapped). Fall back to a modest
# road-class prior for night activity, at low confidence.
ACTIVITY_PRIOR = {
    "primary": 0.35, "primary_link": 0.25, "trunk": 0.25, "trunk_link": 0.2,
    "secondary": 0.3, "secondary_link": 0.25, "tertiary": 0.2,
    "tertiary_link": 0.15, "residential": 0.08, "living_street": 0.08,
    "unclassified": 0.06, "road": 0.06, "pedestrian": 0.2, "service": 0.04,
    "footway": 0.04, "corridor": 0.1, "path": 0.01, "track": 0.0, "steps": 0.02,
}
ISOLATION_PENALTY = 0.5    # fully isolated segment loses this much context
UNDERPASS_PENALTY = 0.3
GUARDIAN_BONUS = 0.1

LOW_SCORE = 0.35           # "exposed" stretch threshold for route reports

# ── Routing ─────────────────────────────────────────────────────────────
WALK_SPEED_MPS = 1.25       # 4.5 km/h
EXPOSURE_BETA = 3.0         # cost = length * (1 + beta * (1 - score))
METRO_SPEED_MPS = 9.0       # ~32 km/h including acceleration
METRO_DWELL_S = 30
METRO_WAIT_S = 6 * 60       # half of a ~12 min late-evening headway
TRANSFER_S = 5 * 60
STATION_ACCESS_M = 120      # entrance to nearest walk node
# Last weekday departures on the Blue and Green lines: 22:30 (Construction
# World, 29 July 2026). Purple and Orange not confirmed; trains leaving a
# terminal at 22:30 pass central stations later, so this is conservative.
METRO_SERVICE_END = "22:30"
METRO_SERVICE_START = "06:00"   # first trains; only matters for after-midnight queries

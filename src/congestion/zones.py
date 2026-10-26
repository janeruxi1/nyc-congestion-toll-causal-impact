"""Taxi-zone groups used throughout the analysis.

The MTA publishes the list of taxi zones inside the Congestion Relief Zone
(Manhattan at and below 60th Street) as a data.ny.gov dataset. This project
re-derives the list from the trip records themselves, which keeps the
pipeline self-contained and checkable (scripts/derive_cbd_zones.py writes
data/reference/cbd_zone_evidence.csv): in January-March 2025, an Uber/Lyft
trip that starts and ends in the same zone carries the CBD fee 90-100% of the
time if the zone is inside the toll zone, and 0-3.5% of the time if it is
not. Six Manhattan zones on the 60th Street boundary show 14-47%; they are
kept out of both groups ("border", the donut) so that trips which may or may
not pay do not blur the contrast.
"""
from __future__ import annotations

# Inside the Congestion Relief Zone (38 taxi zones).
CBD_ZONES: frozenset[int] = frozenset({
    4, 12, 13, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137,
    144, 148, 158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231,
    232, 233, 234, 246, 249, 261,
})

# On or next to the 60th Street boundary: partly tolled, so excluded from the
# comparison ("donut"). 43 Central Park, 140/141 Lenox Hill, 142/143 Lincoln
# Square, 237 Upper East Side South.
BORDER_ZONES: frozenset[int] = frozenset({43, 140, 141, 142, 143, 237})

AIRPORT_ZONES: frozenset[int] = frozenset({1, 132, 138})  # EWR, JFK, LaGuardia
UNKNOWN_ZONES: frozenset[int] = frozenset({264, 265})

# Manhattan zone ids (from data/reference/taxi_zone_lookup.csv), used to tell
# the rest of Manhattan from the outer boroughs. Islands in the harbour
# (103-105 Governor's/Ellis/Liberty) count as Manhattan in the lookup.
MANHATTAN_ZONES: frozenset[int] = frozenset({
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
    103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142,
    143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194,
    202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
    244, 246, 249, 261, 262, 263,
})

GROUPS = ("cbd", "border", "manhattan_north", "outer", "airport", "unknown")


def zone_group(zone: int | None) -> str:
    """Analysis group of a pickup or drop-off zone."""
    if zone is None or zone in UNKNOWN_ZONES or not 1 <= zone <= 263:
        return "unknown"
    if zone in CBD_ZONES:
        return "cbd"
    if zone in BORDER_ZONES:
        return "border"
    if zone in AIRPORT_ZONES:
        return "airport"
    if zone in MANHATTAN_ZONES:
        return "manhattan_north"
    return "outer"


def group_case_sql(col: str) -> str:
    """The same grouping as `zone_group`, as a SQL CASE expression."""
    def ids(s):
        return ", ".join(str(z) for z in sorted(s))
    return (
        f"CASE WHEN {col} IS NULL OR {col} NOT BETWEEN 1 AND 263 THEN 'unknown' "
        f"WHEN {col} IN ({ids(CBD_ZONES)}) THEN 'cbd' "
        f"WHEN {col} IN ({ids(BORDER_ZONES)}) THEN 'border' "
        f"WHEN {col} IN ({ids(AIRPORT_ZONES)}) THEN 'airport' "
        f"WHEN {col} IN ({ids(MANHATTAN_ZONES)}) THEN 'manhattan_north' "
        "ELSE 'outer' END"
    )

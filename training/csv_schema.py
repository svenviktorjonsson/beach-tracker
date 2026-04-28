"""Stable CSV column names for training exports (see HANDOVER_NN.md)."""

# One row per labeled frame (authoritative list for training).
MANIFEST_COLUMNS = (
    "stem",
    "image_filename",
    "image_width",
    "image_height",
    "annotation_source",
)

BALLS_COLUMNS = (
    "stem",
    "image_path",
    "x",
    "y",
    "w",
    "h",
    "scope",
)

PEOPLE_COLUMNS = (
    "stem",
    "image_path",
    "x",
    "y",
    "w",
    "h",
    "role",
)

# Long format: one row per polyline vertex.
COURT_POLYLINE_COLUMNS = (
    "stem",
    "image_path",
    "scope",
    "polyline_id",
    "point_index",
    "x",
    "y",
)

NET_POLYLINE_COLUMNS = COURT_POLYLINE_COLUMNS

EXCLUSION_COLUMNS = (
    "stem",
    "image_path",
    "zone_id",
    "point_index",
    "x",
    "y",
)

CSV_FILES = (
    "images_manifest.csv",
    "balls.csv",
    "people.csv",
    "court_polylines.csv",
    "net_polylines.csv",
    "exclusion_zones.csv",
)

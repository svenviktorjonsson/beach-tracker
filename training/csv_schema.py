"""Stable CSV column names for training exports (see HANDOVER_NN.md)."""

# One row per labeled frame (authoritative list for training).
MANIFEST_COLUMNS = (
    "stem",
    "image_filename",
    "image_width",
    "image_height",
    "annotation_source",
    "distortion_k1",
    "distortion_k2",
    "distortion_k3",
    "rotation_deg",
    "camera_model_version",
    "camera_estimate_status",
    "camera_confidence",
    "camera_fx",
    "camera_fy",
    "camera_cx",
    "camera_cy",
    "camera_k1",
    "camera_k2",
    "camera_k3",
    "camera_p1",
    "camera_p2",
    "camera_yaw_deg",
    "camera_pitch_deg",
    "camera_roll_deg",
    "camera_tx_m",
    "camera_ty_m",
    "camera_tz_m",
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

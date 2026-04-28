"""Label indices aligned with `server.py` scopes/roles."""

# Faster R-CNN: label 0 is background (not assigned to boxes). Foreground 1..N-1.
DETECTION_CLASS_NAMES = (
    "ball_in_play",
    "ball_other",
    "player",
    "referee",
    "other_person",
)

# Segmentation: 0 = background; polylines + exclusion from annotation JSON.
SEGMENTATION_CLASS_NAMES = (
    "court_main",
    "court_adjacent",
    "net_main",
    "net_adjacent",
    "exclusion",
)

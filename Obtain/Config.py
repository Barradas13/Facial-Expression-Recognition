from typing import *

CONFIG: Dict[str, Any] = {
    # --- Image Configs ---
    "image_dir": r"./data",           # image input directory
    "output_dir": "./outputs",              # marked images directory
    "extensions": (".jpg", ".jpeg", ".png", ".bmp"), # compatible image extensions

    # --- FaceMesh (mediapipe==0.10.21, solutions API) ------------------------
    "static_image_mode": True,              # still images, not video tracking
    "max_num_faces": 2,                     # >1 so the largest-face heuristic can choose (D4)
    "refine_landmarks": True,               # adds 10 iris landmarks (478 total vs 468)
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,         # unused in static mode; kept for API parity
    # --- ingestion-time validation (D6) --------------------------------------
    "blur_threshold": 10.0,                # min Laplacian variance
    "brightness_low": 40.0,                 # mean gray below => too dark
    "brightness_high": 220.0,               # mean gray above => too bright
    "min_resolution": 32,                   # min(width, height) in px
    # --- crop / alignment ------------------------------------------------------
    "crop_margin": 0.20,                    # fraction of bbox size padded around landmarks
    "align_output_size": (224, 224),        # canonical crop size (CNN-friendly)
    "save_crops": False,                    # opt-in: writing crops is slow on big datasets
    # --- landmark coordinate table -------------------------------------------
    "include_z": False,                     # D1: 2D by default
    "landmark_indices": None,               # None => all FaceMesh landmarks
}

# Three-class problem; aliases let teams keep their existing folder/file names.
CLASS_NAMES: Tuple[str, ...] = ("smile", "neutral", "sad")
CLASS_ALIASES: Dict[str, str] = {
    "smile": "smile", "smiling": "smile", "happy": "smile", "sorriso": "smile", "sorrindo": "smile",
    "neutral": "neutral", "neutro": "neutral", "neutra": "neutral",
    "sad": "sad", "sorrow": "sad", "triste": "sad",
}
# Filename tokens that should NOT be mistaken for a subject id.
GENERIC_SUBJECT_TOKENS = {"img", "image", "images", "frame", "photo", "foto", "sample", "test", "face", "crop"}

# ---------------------------------------------------------------------------
# FaceMesh landmark index constants — no magic numbers below this cell.
# Naming is ANATOMICAL (D9): the subject's right eye (33-group) appears
# image-left in non-mirrored photos.
# ---------------------------------------------------------------------------
# 6-point Eye Aspect Ratio rings. Correspondence with the dlib 68-point model
# (which `padroes/equacoes.py::razaoDistOlhos` was written for):
#   right eye: dlib 36,37,38,39,40,41  ->  FaceMesh 33,160,158,133,153,144
#   left  eye: dlib 42,43,44,45,46,47  ->  FaceMesh 362,385,387,263,373,380

EYE_RIGHT: Dict[str, int] = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
EYE_LEFT: Dict[str, int] = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 373, "p6": 380}
EYE_RIGHT_IDXS: Tuple[int, ...] = tuple(EYE_RIGHT.values())
EYE_LEFT_IDXS: Tuple[int, ...] = tuple(EYE_LEFT.values())
ALL_EYE_IDXS: Tuple[int, ...] = EYE_RIGHT_IDXS + EYE_LEFT_IDXS

MOUTH_CORNER_RIGHT = 61    # image-left corner
MOUTH_CORNER_LEFT = 291    # image-right corner
LIP_UPPER_INNER = 13
LIP_LOWER_INNER = 14
LIP_UPPER_OUTER = 0
LIP_LOWER_OUTER = 17

NOSE_TIP = 1
CHIN_BOTTOM = 152
FOREHEAD_TOP = 10
FACE_SIDE_RIGHT = 234      # image-left face-oval boundary
FACE_SIDE_LEFT = 454       # image-right face-oval boundary

BROW_RIGHT = 105           # mid upper-brow point, subject's right
BROW_LEFT = 334            # mid upper-brow point, subject's left

IRIS_RIGHT_CENTER = 468    # present only when refine_landmarks=True
IRIS_LEFT_CENTER = 473

# Every landmark index referenced by the geometric features (bounds-checking).
MAX_USED_INDEX = max(*EYE_RIGHT_IDXS, *EYE_LEFT_IDXS, MOUTH_CORNER_RIGHT, MOUTH_CORNER_LEFT,
                     LIP_UPPER_INNER, LIP_LOWER_INNER, LIP_UPPER_OUTER, LIP_LOWER_OUTER,
                     NOSE_TIP, CHIN_BOTTOM, FOREHEAD_TOP, FACE_SIDE_RIGHT, FACE_SIDE_LEFT,
                     BROW_RIGHT, BROW_LEFT)

# Canonical column order for the geometric table (schema stability).
GEOMETRIC_FEATURE_ORDER: Tuple[str, ...] = (
    "ear_left", "ear_right", "mar", "mouth_width", "mouth_height",
    "brow_eye_distance_left", "brow_eye_distance_right", "mouth_curvature",
    "nose_to_mouth_corner_left", "nose_to_mouth_corner_right",
    "face_width", "face_height",
)
GEOMETRIC_META_COLUMNS: Tuple[str, ...] = (
    "image_path", "source", "subject_id", "label", "face_found", "n_faces",
)
GEOMETRIC_COLUMNS: Tuple[str, ...] = GEOMETRIC_META_COLUMNS + ("iod_px",) + GEOMETRIC_FEATURE_ORDER + ("error",)

MANIFEST_COLUMNS: Tuple[str, ...] = (
    "image_path", "source", "subject_id", "label",
    "width", "height", "face_found", "n_landmarks", "blur_score", "brightness", "rejected_reason",
)
REJECTED_COLUMNS: Tuple[str, ...] = (
    "image_path", "source", "subject_id", "label",
    "width", "height", "blur_score", "brightness", "rejected_reason",
)
LANDMARK_META_COLUMNS: Tuple[str, ...] = (
    "image_path", "source", "subject_id", "label", "face_found", "n_landmarks", "error",
)

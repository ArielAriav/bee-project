# config.py

MODEL_PATH = r"runs\tag_detection\train\weights\best.pt"
VIDEO_PATH = r"data\raw\entrance\beeVideo.mp4"

DET_CONF = 0.4
PADDING = 6
UPSCALE = 4
MAX_DIGITS = 3
DISPLAY_MIN_CONF = 0.70

OCR_EVERY = 4
SKIP_IF_CONF_GE = 0.95

BLUR_MIN_LOCK = 40
BLUR_MIN_FIND = 30
CROP_CENTER_RATIO = 0.7

ANGLES = (0, -10, 10, -20, 20, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90)

VOTE_HISTORY = 12
VOTE_DOMINANCE = 0.60
MIN_ACCEPT_CONF = 0.45

OCR_PARAMS = dict(
    detail=1,
    allowlist="0123456789",
    paragraph=False,
    text_threshold=0.6,
    low_text=0.3,
    link_threshold=0.4,
    contrast_ths=0.05,
    adjust_contrast=0.7,
)

COLOR = (0, 0, 255)

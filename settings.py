import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL = "weights/best.pt"
SEGMENTATION_MODEL = "weights/best.pt"
SOURCES_LIST = ["Image", "Video", "Webcam", "RTSP", "YouTube"]
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
YOUTUBE = "YouTube"

DEFAULT_IMAGE = os.path.join(BASE_DIR, "default.jpg")
DEFAULT_DETECT_IMAGE = os.path.join(BASE_DIR, "default_detect.jpg")

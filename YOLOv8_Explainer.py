from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam import GradCAM
import torch

def yolov8_heatmap(weight, conf_threshold=0.25, method="EigenCAM", layer=None, ratio=0.02, show_box=False, renormalize=False):
    model = YOLO(weight)

    def custom_predict(img_path=0):
        result = model.predict(
            img_path, conf=conf_threshold, show=False, save=False
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img) / 255.0

        # Dummy CAM overlay for demonstration
        cam = np.zeros_like(img)
        cam[:, :, 0] = 1  # red heatmap

        cam_output = (img * 0.5 + cam * 0.5)
        cam_output = (cam_output * 255).astype(np.uint8)

        return [cam_output]

    return custom_predict

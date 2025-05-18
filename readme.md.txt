# Coconut Crop Disease Detection 🌴🦠

This is a Streamlit-based deep learning app to detect diseases in coconut trees using YOLOv8 and Grad-CAM for explainability.

## 🚀 Features

- Upload an image of a coconut tree leaf
- Detect if any disease is present using a YOLOv8 model
- Highlight the affected area using Grad-CAM heatmaps
- Uses Google Generative AI (Gemini) for additional feedback

## 🧠 Tech Stack

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- Grad-CAM (pytorch-grad-cam)
- Google Generative AI
- OpenCV, Pillow, Torch

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py

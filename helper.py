from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def image_detection(conf, model, uploaded_image):
    results = model.predict(uploaded_image, conf=conf)
    return results

def play_stored_video(conf, model):
    pass

def play_webcam(conf, model):
    pass

def play_rtsp_stream(conf, model):
    pass

def play_youtube_video(conf, model):
    pass

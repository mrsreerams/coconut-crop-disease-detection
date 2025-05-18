# Python In-built packages
from pathlib import Path
import PIL
import google.generativeai as genai
# External packages
import streamlit as st
import os 
# Local Modules
import settings
import helper
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from YOLOv8_Explainer import yolov8_heatmap
import tempfile


model2 = yolov8_heatmap(
    weight="weights/best.pt", 
        conf_threshold=0.4, 
        method = "EigenCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        ratio=0.02,
        show_box=True,
        renormalize=False,
)

# Load the YOLOv8 model
modell = YOLO("weights/best.pt")

# Function to display heatmap with textual explanations and confidence levels
def yolov8_explainable_predictions(model2, img_path):
    # Get the original image and the heatmap
    imagelist = model2(img_path=img_path)

    # Convert the heatmap to a NumPy array if it's not already
    image_with_heatmap = np.array(imagelist[0], dtype=np.uint8)

    # Perform inference using YOLOv8 model
    results = modell.predict(img_path)

    # Access the Boxes object which contains the bounding boxes and confidence scores
    boxes = results[0].boxes  # Get bounding boxes from results
    class_names = results[0].names  # Get class names from results

    # Prepare textual explanation list
    explanations = []

    # Iterate over the boxes to get the confidence scores and class labels
    for box in boxes:
        # Extract bounding box coordinates, confidence score, and class index
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class index

        # Get class name from the class index
        class_name = class_names[class_id]

        # Convert confidence to percentage
        confidence_text = f"{class_name}: {confidence * 100:.2f}%"

        # Draw bounding box on the image
        cv2.rectangle(image_with_heatmap, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

        # Put confidence text above bounding box
        cv2.putText(image_with_heatmap, confidence_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Blue text

        # Create textual explanation for this detection
        explanation = f"The model detected '{class_name}' with {confidence * 100:.2f}% confidence. " \
                      f"The highlighted regions (see heatmap) influenced this decision, indicating {class_name}."
        explanations.append(explanation)

    # Display the image with heatmap and confidence scores
    # plt.imshow
    image_rgb = (cv2.cvtColor(image_with_heatmap, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    st.image(image_rgb)
    # plt.show()

    # Display textual explanations
    st.info("\nModel Explanations:")
    for explanation in explanations:
        st.info(explanation)


# Setting page layout
st.set_page_config(
    page_title="Coconut Disease Detection",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Coconut Disease Detection")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 10, 100, 10)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
            
        else:
            if st.sidebar.button('Detect Objects'):
                # Save the uploaded image as a temporary JPEG file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    uploaded_image.save(temp_file, format="JPEG")
                    temp_file_path = temp_file.name
                # Call the function with explainable predictions
                yolov8_explainable_predictions(model2, temp_file_path)
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                names = res[0].names
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                class_detections_values = []
                for k, v in names.items():
                    class_detections_values.append(res[0].boxes.cls.tolist().count(k))
                # create dictionary of objects detected per class
                classes_detected = dict(zip(names.values(), class_detections_values))
                # st.text(classes_detected)
                if 'bud root dropping' in classes_detected:
                    disease_detected = classes_detected['bud root dropping']
                    if disease_detected:
                        if not disease_detected == 0:
                            # cause = inputdisease(disease_detected="bud root dropping")
                            # st.info(f"ExplainableAI : In this Tree, {cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[BUD ROOT DROPPING] : {disease_detected}")
                            st.warning(f"#### [Cause]: \n Related to poor root health, inadequate nutrients, or environmental stress")
                            st.success(f"#### [Solution]: \n 1. Conduct systematic inspections of the coconut trees to spot early signs of disease or pest infestations. \n 2.  Regularly test soil to determine nutrient needs and adjust fertilization accordingly. \n 3. Implement efficient irrigation systems to avoid water stress and waterlogging. \n 4. Employ integrated pest management (IPM) strategies to control pests that can act as disease vectors.")
                            st.link_button("Go to Product Link","https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide")
                if 'bud rot' in classes_detected:
                    disease_detected = classes_detected['bud rot']
                    if disease_detected:
                        if not disease_detected == 0:
                            # cause = inputdisease(disease_detected="bud rot")
                            # st.info(f"ExplainableAI : In this Tree, {cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[BUD ROT] : {disease_detected}")
                            st.warning(f"#### [Cause] : \n Bud rot is caused by the fungus Phytophthora palmivora. It leads to black lesions on young fronds and leaves, weakening the tree.")
                            st.success(f"#### [Soultion] : \n 1. Regularly inspect your coconut trees for signs of black lesions \n 2. Avoid water stress and waterlogging. \n 3. Some of the more common coconut tree disease issues include fungal or bacterial problems")
                            st.link_button("Go to Product Link","https://cultree.in/products/shivalik-roksin-thiophanate-methyl-70-wp-fungicide")
                if 'gray leaf spot' in classes_detected:
                    disease_detected = classes_detected['gray leaf spot']
                    if disease_detected:
                        if not disease_detected == 0:
                            # cause = inputdisease(disease_detected="gray leaf spot")
                            # st.info(f"ExplainableAI : In this Tree, {cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[GRAY LEAF SPOT] : {disease_detected}")
                            st.warning(f"#### [Cause] \n Gray leaf spots are caused by both fungi and bacteria. Circular or elongated spots develop on foliage.")
                            st.success(f"#### [Soultion] \n 1. Pestalotiopsis palmarum is the primary causative agent of gray leaf spot. It affects not only coconut trees but also bananas and date palms. The fungus leads to leaf spots, petiole/rachis blights, and sometimes bud rot in palms. \n 2. Gray leaf spot tends to be more severe on older leaves. Unfavorable growing conditions, such as excessive moisture or poor ventilation, can exacerbate the disease. \n 3. Rain and wind play a crucial role in spreading the spores of both brown leaf spot and gray leaf spot fungi. Consequently, these diseases are more common during wet weather.")
                            st.link_button("Go to Product Link","https://www.bighaat.com/products/agriventure-cooxy")
                if 'leaf rot' in classes_detected:
                    disease_detected = classes_detected['leaf rot']
                    if disease_detected:
                        if not disease_detected == 0:
                            # cause = inputdisease(disease_detected="leaf rot")
                            # st.info(f"ExplainableAI : In this Tree, {cause}")
                            st.info(f"#### Disease Detected: \n ##### :blue[LEAF ROT] : {disease_detected}")
                            st.warning(f"#### [Cause] \n This type of disease is majorly caused by fungi, Peastalozzia palmarum, and the Bipolaris incurvata. Initially, there will be a visible appearance of the yellow-brown spots that appear on top of the leaflets from the lower fronds. This then gradually enlarges and tends to turn grey on the coconut planting.")
                            st.success(f"#### [Soultion] \n 1. Inspect your coconut tree during wet weather, as rain and wind can spread spores that cause leaf spots. \n 2. Apply fungicides to control leaf spot diseases \n 3. If your coconut tree exhibits a condition known as “pencil point disorder,” it may be due to a lack of micronutrients.")
                            st.link_button("Go to Product Link","https://ariesagro.com/jahaan-hexaconazole5-w-w/")
                if 'stembleeding' in classes_detected:
                    disease_detected = classes_detected['stembleeding']
                    if disease_detected:
                        if not disease_detected == 0:
                            # cause = inputdisease(disease_detected="stembleeding")
                            # st.info(f"ExplainableAI : In this Tree, {cause}")
                            st.title(f"#### Disease Detected: ##### \n :blue[STEMBLEEDING] : {disease_detected}")
                            st.warning(f"#### [Cause] \n Stem bleeding disease of coconut, caused by Thielaviopsis paradoxa (de Seyness) Von Hohnel, is widely prevalent in all coconut growing regions in the tropics.")
                            st.success(f"#### [Soultion] \n 1. Chisel out the affected tissues completely. \n 2. Paint the wound with Bordeaux paste or apply coal tar after 1-2 days. \n 3. Apply neem cake (approximately 5 kg per palm) to the basin along with other organic materials. \n 4. Root feed with Tridemorph (Calixin-5%) in water (5 ml in 100 ml) thrice a year during April-May, September-October, and January-February.")
                            st.link_button("Go to Product 1 Link","https://www.bighaat.com/products/blue-copper-fungicide-1")
                            st.link_button("Go to Product 2 Link","http://www.rayfull.com/Productshows.asp?ID=338")
                            st.link_button("Go to Product 3 Link","https://krishisevakendra.in/products/bordeaux-mixture")
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

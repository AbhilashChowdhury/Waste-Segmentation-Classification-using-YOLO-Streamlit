import streamlit as st
from PIL import Image
import os
import uuid
from ultralytics import YOLO
import shutil

# Directories
UPLOAD_DIR = os.path.join("predicts", "uploaded_images")  # Under 'predicts/'
OUTPUT_DIR = os.path.join("predicts", "output")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 segmentation model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the same folder

# Streamlit title
st.title("♻️ Waste Segmentation using YOLO")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Save uploaded image
    file_ext = uploaded_file.name.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, unique_filename)
    image.save(uploaded_path)

    # Run YOLOv8 segmentation model
    results = model(uploaded_path, project="predicts", name="output", save=True, exist_ok=True)

    # Extract detected class names
    classes = results[0].names
    labels = set([classes[int(cls)] for cls in results[0].boxes.cls])

    # Find the annotated (output) image
    annotated_path = os.path.join("predicts", "output", os.path.basename(uploaded_path))

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("🖼 Segmented Output")
        if os.path.exists(annotated_path):
            annotated_img = Image.open(annotated_path)
            st.image(annotated_img, caption="YOLOv8 Segmentation", use_container_width=True)
        else:
            st.warning("Annotated image not found.")

    # Prediction result below both images
    st.subheader("🧠 Model Prediction")
    st.success("Detected: " + ", ".join(labels))

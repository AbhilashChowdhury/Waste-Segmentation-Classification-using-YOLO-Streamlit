import streamlit as st
from PIL import Image
import os
import uuid
from ultralytics import YOLO
import glob
import shutil

# Directories
UPLOAD_DIR = os.path.join("predicts", "uploaded_images")  # Now under 'predicts/'
OUTPUT_DIR = os.path.join("predicts", "output")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 segmentation model (update path if needed)
model = YOLO("best.pt")  # Make sure best.pt is in the same folder or provide full path

# Streamlit title
st.title("♻️ Waste Segmentation using YOLOv8")

# Upload file
uploaded_file = st.file_uploader("📤 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and convert if needed
    image = Image.open(uploaded_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Save uploaded image
    file_ext = uploaded_file.name.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, unique_filename)
    image.save(uploaded_path)

    # Display uploaded image
    st.subheader("📷 Uploaded Image")
    st.image(image, caption="Original Image", use_container_width=True)

    # Run YOLOv8 segmentation model and save results in custom folder
    results = model(uploaded_path, project="predicts", name="output", save=True, exist_ok=True)

    # Display detected class (placeholder — you can extract from results)
    st.subheader("🧠 Model Prediction")
    classes = results[0].names
    labels = set([classes[int(cls)] for cls in results[0].boxes.cls])
    st.success("Detected: " + ", ".join(labels))

    # Display annotated image from predicts/output
    st.subheader("🖼 Segmented Output")
    
    # Find the annotated image file
    annotated_path = os.path.join("predicts", "output", os.path.basename(uploaded_path))
    if os.path.exists(annotated_path):
        annotated_img = Image.open(annotated_path)
        st.image(annotated_img, caption="YOLOv8 Segmentation", use_container_width=True)
    else:
        st.warning("Annotated image not found.")

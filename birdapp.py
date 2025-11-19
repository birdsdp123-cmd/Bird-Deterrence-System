import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO

# Load YOLO model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Adjust the path if necessary

def annotate_birds(image_path):
    """Use YOLO to detect and annotate birds"""
    img = cv2.imread(image_path)
    results = model(img)
    annotated = img.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == "bird":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert to RGB for Pillow and return
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated_rgb)
    return annotated_img

# Streamlit UI
st.title("Bird Annotation Tool")
st.write("Upload an image to detect and annotate birds.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Annotate the image
    annotated_image = annotate_birds("uploaded_image.jpg")
    
    # Display the original and annotated images
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
    
    # Allow users to download the annotated image
    st.download_button("Download Annotated Image", data=annotated_image, file_name="annotated_image.jpg", mime="image/jpeg")



import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


# ------------------ Preprocessing Functions ------------------ #

def apply_clahe_defog(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except:
        return image

def simple_white_balance(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        a = a - ((avg_a - 128) * (l / 255.0) * 1.1)
        b = b - ((avg_b - 128) * (l / 255.0) * 1.1)
        result = cv2.merge((l, a, b))
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    except:
        return image

def adjust_gamma(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_contrast_brightness(image, contrast=1.1, brightness=5):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def sharpen_image(image, strength=1.2):
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1 + strength, blur, -strength, 0)

def warm_tone(image, intensity=0.18):
    b, g, r = cv2.split(image)
    r = cv2.add(r, int(30 * intensity))
    b = cv2.subtract(b, int(30 * intensity))
    return cv2.merge((b, g, r))

def apply_vignette(image, strength=0.45):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * strength)
    kernel_y = cv2.getGaussianKernel(rows, rows * strength)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        output[:, :, i] = image[:, :, i] * mask
    return output.astype(np.uint8)

def preprocess_pipeline(image):
    img = apply_clahe_defog(image)
    img = simple_white_balance(img)
    img = adjust_gamma(img, gamma=1.2)
    img = adjust_contrast_brightness(img, contrast=1.1, brightness=5)
    img = sharpen_image(img, 1.2)
    img = warm_tone(img, 0.18)
    img = apply_vignette(img, 0.45)
    img = adjust_contrast_brightness(img, contrast=1.05, brightness=-12)
    return img

# ------------------ Streamlit UI ------------------ #

st.title("🔍 YOLOv8s vs Distilled Model Comparison (With & Without Preprocessing)")

uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} images uploaded!")

    # Load Models Once
    model_default = YOLO("yolov8s.pt").to("cuda")
    model_custom = YOLO("best.pt").to("cuda")


    for file in uploaded_files:
        st.subheader(f"Image: {file.name}")

        # Read Original Image
        image = np.array(Image.open(file).convert("RGB"))
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Preprocessed Image
        processed_img = preprocess_pipeline(bgr_image)

        # YOLOv8s: No Preprocessing
        res_nopre = model_default.predict(bgr_image, conf=0.5, verbose=False)[0].plot()

        # YOLOv8s: With Preprocessing
        res_pre = model_default.predict(processed_img, conf=0.5, verbose=False)[0].plot()

        # Custom Model (Always Preprocessed)
        res_custom = model_custom.predict(processed_img, conf=0.5, verbose=False)[0].plot()

        # Convert to Display (RGB)
        res_nopre = cv2.cvtColor(res_nopre, cv2.COLOR_BGR2RGB)
        res_pre = cv2.cvtColor(res_pre, cv2.COLOR_BGR2RGB)
        res_custom = cv2.cvtColor(res_custom, cv2.COLOR_BGR2RGB)

        # Show Outputs Side-by-Side
        col1, col2, col3 = st.columns(3)

        col1.image(res_nopre, caption="YOLOv8s (No Preprocessing)")
        col2.image(res_pre, caption="YOLOv8s (With Preprocessing)")
        col3.image(res_custom, caption="My Model best.pt (With Preprocessing)")

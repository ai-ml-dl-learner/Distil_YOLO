import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")



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
    print("   → Preprocessing (Defog + WB + Gamma + Contrast + Sharpen + Evening Style)")
    img = apply_clahe_defog(image)
    img = simple_white_balance(img)
    img = adjust_gamma(img, gamma=1.2)
    img = adjust_contrast_brightness(img, contrast=1.1, brightness=5)
    img = sharpen_image(img, strength=1.2)

    # Evening Look (Warm + Vignette + Slight Darken)
    img = warm_tone(img, intensity=0.18)
    img = apply_vignette(img, strength=0.45)
    img = adjust_contrast_brightness(img, contrast=1.05, brightness=-12)

    return img



MODEL_PATH = "best.pt"
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "my_model_output1"



if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n Output Folder: {OUTPUT_FOLDER}")

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        exit()

    print(f"⚡ Loading YOLO Model on GPU...")
    model = YOLO(MODEL_PATH).to("cuda")   # GPU Enabled

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_paths:
        print(f"❌ No Images Found in: {INPUT_FOLDER}")
        exit()

    print(f"✅ {len(image_paths)} images detected. Starting...\n")

    for img_path in image_paths:
        print(f"🔹 Processing: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            print("   ⚠ Skipping (Image could not be read)")
            continue

        enhanced = preprocess_pipeline(img)

        print("   → YOLO Detection Running...")
        results = model.predict(enhanced, conf=0.5, verbose=False)
        annotated = results[0].plot()

        save_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
        cv2.imwrite(save_path, annotated)

        print(f" Saved Output → {save_path}\n")

    print("\n Processing Completed Successfully.\n")

import os
from ultralytics import YOLO
from glob import glob

MODEL_PATH = 'yolo11n.pt'

INPUT_FOLDER = 'images'  
OUTPUT_FOLDER = 'yolo_model_output'

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output will be saved to '{OUTPUT_FOLDER}'")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
    else:
        print(f"Loading model from '{MODEL_PATH}'...")
        model = YOLO(MODEL_PATH)
        print(f"Running batch detection on all images in '{INPUT_FOLDER}'...")
        results = model.predict(
            source=INPUT_FOLDER,
            save=True,
            project=OUTPUT_FOLDER,
            name='detection_run', 
            exist_ok=True, 
            conf=0.5 
        )
        print("\nBatch detection complete.")
        print(f"Annotated images have been saved in '{os.path.join(OUTPUT_FOLDER, 'detection_run')}'")
        processed_files = 0
        for result in results:
            processed_files += 1
        
        print(f"Processed {processed_files} image(s).")

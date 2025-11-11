# High-Fidelity Knowledge Distillation for Object Detection

This repository contains the complete implementation of a computer vision project focused on creating a highly efficient yet accurate object detection model. The core of this project is the successful application of **feature-based, heterogeneous knowledge distillation** to transfer the deep contextual understanding of a large Transformer-based model (RT-DETR-x) into a lightweight, fast CNN-based model (YOLOv8s).

## Abstract

In real-world computer vision applications, there exists a fundamental tension between model accuracy and deployment feasibility. Large models offer high precision but are computationally expensive, while small models are fast but often less accurate. This project bridges that gap by "distilling" the knowledge from a large teacher model into a compact student model. The resulting model (`best.pt`) achieves a strong mAP50 score of **56.5%** while maintaining a lean parameter count of just **11.1 million**, making it ideal for high-performance, real-time scenarios. This repository includes the complete training code, a sophisticated image preprocessing pipeline, and an interactive Streamlit application for live demonstration and comparison.

## Project Architecture

The project is divided into two primary phases: the **Training Phase** and the **Inference & Demonstration Phase**.

### 1. The Training Architecture: Knowledge Distillation

The training process is orchestrated by a custom `DetectionTrainer` built on the Ultralytics framework. At each training step, the following occurs:

1.  **Parallel Input**: An input image is simultaneously fed to the expert "Teacher" model (RT-DETR-x, frozen in evaluation mode) and the "Student" model (YOLOv8s, in training mode).
2.  **Feature Extraction**: PyTorch hooks are attached to intermediate layers of both models to capture their high-level feature maps—a snapshot of their internal "understanding" of the image.
3.  **Feature Adaptation**: Since the teacher and student have different architectures, their feature maps have different dimensions. A learnable **Adapter** (a 1x1 Convolutional layer) is used to translate the student's features into the same "language" as the teacher's.
4.  **Dual Loss Calculation**: The student's learning is guided by a composite loss function:
    *   **Detection Loss**: Measures the accuracy of the student's final predictions against the ground-truth labels (using Box, Class, and DFL losses).
    *   **Distillation Loss**: A Mean Squared Error (MSE) loss that measures the similarity between the student's adapted features and the teacher's features.
5.  **Backpropagation**: The total combined loss is backpropagated to update the weights of **only the student model and the adapter layer**, effectively "teaching" the student to think like the expert teacher.

### 2. The Inference Architecture: Streamlit Demonstration App

To showcase the results, an interactive web application (`app_with_diagram.py`) was developed using Streamlit. Its workflow is as follows:

1.  **Image Upload**: The user uploads an image.
2.  **Preprocessing Pipeline**: The image is passed through a multi-step enhancement pipeline (CLAHE, white balance, gamma correction, sharpening, etc.) to normalize and improve image quality.
3.  **Parallel Inference**: The *enhanced* image is fed to both the standard `yolov8n.pt` model and our custom-distilled `best.pt` model.
4.  **Side-by-Side Visualization**: The application displays the original image, the enhanced image, and a direct comparison of the detections made by both models, clearly demonstrating the superior performance of the distilled model.

## How to Use This Repository

1.  **Clone the Repository**
    ```
    git clone https://github.com/your-username/Distil_YOLO.git
    cd Distil_YOLO
    ```

2.  **Set Up Environment**
    It is recommended to use a virtual environment.
    ```
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all necessary packages.
    ```
    pip install -r requirements.txt
    ```

4.  **Run the Interactive Demo**
    Launch the Streamlit web application.
    ```
    streamlit run app_with_diagram.py
    ```
    Now, open your web browser to the local URL provided by Streamlit to start using the application.



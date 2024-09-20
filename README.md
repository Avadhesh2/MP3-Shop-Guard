Shop Guard: A Suspicious Behavior Detection System
Introduction
Shop Guard is a real-time object detection and classification system designed to detect suspicious activities in retail environments. The system leverages the YOLOv8 model to classify behaviors into two categories: "Normal" and "Suspicious". It can be used for loss prevention, enhancing security, and improving the safety of shop premises.

Table of Contents
Installation
Project Structure
Dataset
Training
Inference
Evaluation
Results
Future Work
License
Installation
To set up the Shop Guard project, follow the instructions below.

Requirements
Python 3.9+
Google Colab or local machine with GPU for training
The following Python packages:
ultralytics (for YOLOv8)
torch
opencv-python
numpy
matplotlib
You can install the required dependencies using pip:
pip install ultralytics torch opencv-python numpy matplotlib
Setting Up the Project
Clone this repository or download the project files.
Download the required dataset from Roboflow or your preferred dataset source and place it in the /dataset folder.
Ensure you have the yolov8n.pt model weights downloaded (or train your model) and placed in the appropriate location.
Project Structure
The project folder is structured as follows:
/shop-guard/
│
├── /dataset/                     # Dataset folder
│   ├── images/train/             # Training images
│   ├── images/val/               # Validation images
│   ├── labels/train/             # Training labels
│   ├── labels/val/               # Validation labels
│   └── data.yaml                 # Dataset configuration
│
├── /runs/                        # Output results (after training and evaluation)
│   ├── detect/                   # Detection results
│   └── train/                    # Training results
│
├── shopguard_inference.py         # Script for running inference
├── shopguard_train.py             # Script for training the model
└── README.md                     # This file
Dataset
The dataset is structured according to YOLOv8 requirements, with images stored in .jpg format and corresponding label files stored in .txt format.

Train Images: dataset/images/train/
Validation Images: dataset/images/val/
Labels: Stored in dataset/labels/ for both train and validation sets.
Classes:
0: Suspicious
1: Normal
The dataset is configured using the data.yaml file:
train: /content/dataset/images/train
val: /content/dataset/images/val

nc: 2
names: ['suspicious', 'normal']
Training
To train the YOLOv8 model on your dataset, run the following code in a Google Colab notebook or a local machine:
from ultralytics import YOLO

# Load YOLOv8 model (can be 'yolov8n.pt' for a small model)
model = YOLO('yolov8n.pt')

# Train the model on the custom dataset
model.train(data='/path_to_your_dataset/data.yaml', epochs=50, batch=16, imgsz=640, name='shop_guard')
The data.yaml file should be updated to point to the dataset directory.

Inference
Once your model is trained, you can run inference on a single image or a batch of images to classify the behavior as "Suspicious" or "Normal".

python
Copy code
from ultralytics import YOLO

# Load the trained model
model = YOLO('/path_to_your_model/best.pt')

# Run inference on a test image
results = model('/path_to_image/test_image.jpg')

# Show results (image with bounding boxes)
results.show()

# Get the class name (Suspicious/Normal)
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        print(f"Detected: {class_name}")
Evaluation
You can evaluate the performance of your model on a validation dataset to get metrics like mAP (mean Average Precision), precision, and recall:

python
Copy code
# Evaluate the model on validation data
metrics = model.val(data='/path_to_dataset/data.yaml')

# Print evaluation metrics
print(metrics)
Results
You can visualize the results using the images saved in the /runs/detect/ or /runs/train/ directories, which will show detected bounding boxes, class labels, and confidence scores.

Training Metrics: Stored in runs/train/
Detection Results: Saved images with detected classes and bounding boxes in runs/detect/
Future Work
Expand Dataset: Increase the dataset size for better generalization.
Add More Classes: Include additional classes for a more detailed classification of shoplifting activities.
Real-time Detection: Implement real-time video processing using OpenCV.
Deploy the Model: Deploy the system in a production environment using a web application or API.

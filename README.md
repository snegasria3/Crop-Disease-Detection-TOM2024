🌿 Explainable Multi-Crop Plant Disease Diagnosis
Using Convolutional Neural Networks on TOM2024 Dataset
📌 Overview

This project presents an Explainable AI-based deep learning system for detecting plant diseases across multiple crops (Tomato, Onion, and Maize) using the TOM2024 dataset.

The system compares three CNN-based architectures and integrates Explainable AI (XAI) techniques to improve transparency and trust in predictions. A Flask web application is developed for real-time disease detection and treatment suggestions.

🚀 Features
🌱 Multi-crop disease classification (30 classes)
🧠 Deep learning models:
Custom CNN (LeafNetCNN)
ResNet-18
EfficientNet-B0 (Best performing)
🔍 Explainability:
Grad-CAM
LIME
🌐 Flask web app for real-time prediction
📊 Performance evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)
🗂️ Dataset
Dataset Used: TOM2024
Contains:
12,000+ labeled images
Tomato, Onion, and Maize leaves
Real-world conditions (lighting, noise, occlusion)
30 disease + healthy classes
🏗️ System Architecture

The system follows a complete pipeline:

Image Acquisition
Preprocessing & Augmentation
Model Training & Inference
Explainability (Grad-CAM & LIME)
Web Deployment (Flask)
🧠 Models Used
1. LeafNetCNN (Custom CNN)
Lightweight architecture
Efficient for real-time prediction
2. ResNet-18
Uses residual connections
Better gradient flow
3. EfficientNet-B0 ⭐
Best performance
Uses compound scaling
Achieved highest accuracy
📊 Results
Model	Test Accuracy	F1 Score
CNN	69.26%	0.51
ResNet-18	82.75%	0.68
EfficientNet-B0	88.31%	0.88
ROC-AUC: 0.96 (EfficientNet-B0)
Statistically significant improvement (p < 0.05)
🔍 Explainability
Grad-CAM
Highlights infected regions on leaves
LIME
Provides pixel-level interpretability

➡️ Ensures model decisions are biologically meaningful

🌐 Web Application
Built using Flask
Features:
Upload leaf image
Disease prediction
Treatment suggestions

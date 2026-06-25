# J.U.S.T AI Projects Portfolio

Welcome to my collection of AI and Machine Learning projects completed during my 4 years of study at **Jordan University of Science and Technology (J.U.S.T)**. This repository showcases my journey in artificial intelligence, computer vision, and deep learning.

---

## 📚 Overview

This portfolio contains projects spanning:
- **Computer Vision** - Object Detection, Image Classification, Real-time Analysis
- **Deep Learning** - YOLO, Neural Networks, Transfer Learning
- **Data Science** - Dataset Annotation, Analysis, Preprocessing
- **Real-time Systems** - Video Processing, Multi-Object Tracking (MOT)
- **Sports Analytics** - Player Detection, Ball Tracking, Possession Metrics

---

## 🎯 Key Project: Jordanian Pro League Football Analysis

### Project Overview
A comprehensive real-time computer vision system for analyzing Jordanian Professional Football League matches using YOLOv8 object detection and multi-object tracking.

### Achievements
- **84.6% mAP@0.5** on custom-labeled dataset
- **98.1% player detection recall** (Team A/Team B differentiation)
- **88.9% referee detection recall**
- **56.2% ball detection recall**
- Real-time inference at **0.7ms preprocess, 12.3ms inference** per image

### Technologies
- **Detection**: YOLOv8 (960px input size)
- **Tracking**: Multi-Object Tracking (MOT)
- **Dataset**: Custom-annotated using CVAT (4,280 train, 1,462 val, 720 test images)
- **Deployment**: Real-time pipeline with ball possession calculation

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Training Images | 4,280 |
| Validation Images | 1,462 |
| Test Images | 720 |
| Total Annotated | 6,462 |
| Classes | 3 (Player, Ball, Referee) |
| Total Objects | 51,659 |

### Model Performance - Test Set

**Overall Metrics:**
- mAP@0.5: 84.59%
- mAP@0.5-0.95: 42.02%
- Mean Precision: 88.55%
- Mean Recall: 81.08%

**Per-Class Performance:**

| Class | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|--------------|
| Player | 93.05% | 97.21% | 97.24% | 56.80% |
| Ball | 81.31% | 57.10% | 63.68% | 25.02% |
| Referee | 91.27% | 88.95% | 92.84% | 44.23% |

### Error Analysis

**Detection Recalls (by team):**
- Team A Detection Recall: **98.17%**
- Team B Detection Recall: **98.04%**
- Overall Player Recall: **98.10%**

**Confusion Analysis:**
- Referee wrongly classified as Player: 14.61%
- Player wrongly classified as Referee: 0.16%
- Ball Detection Recall: 56.16%

### Threshold Analysis
The model's performance across different confidence thresholds:

| Threshold | Player Recall | Ball Recall | Referee Recall | Referee→Player Rate |
|-----------|---------------|-------------|-----------------|-------------------|
| 0.10 | 98.64% | 68.02% | 89.58% | 20.46% |
| 0.15 | 98.50% | 63.34% | 89.20% | 18.42% |
| 0.20 | 98.23% | 59.44% | 89.07% | 16.65% |
| **0.25** | **98.10%** | **56.16%** | **88.95%** | **14.61%** |
| 0.30 | 97.93% | 49.77% | 88.56% | 12.96% |
| 0.40 | 97.58% | 31.36% | 86.66% | 10.17% |

### Project Structure
```
Football-Analysis/
├── Before_Training/
│   └── before_training.ipynb          # Baseline HuggingFace model evaluation
├── After_Training/
│   └── Best_Results.ipynb             # Trained model evaluation & analysis
├── Models/
│   └── best.pt                        # Trained YOLOv8 model (21.5MB)
├── Results/
│   ├── three_class_swapped_summary.csv
│   ├── three_class_swapped_per_class.csv
│   ├── swapped_error_analysis_summary.csv
│   ├── swapped_threshold_comparison.csv
│   ├── swapped_image_error_analysis.csv
│   └── confusion_matrices/
└── README.md
```

---

## 📊 Training Details

### Data Preparation
- **Original Dataset**: 4-class labels (Player Team A, Player Team B, Ball, Referee)
- **Conversion**: Merged team-specific players into single "Player" class (3-class)
- **Augmentation**: Light oversampling of minority classes (ball, referee)
- **Final Training Set**: 10,858 weighted entries from 4,280 unique images

### Model Configuration
- **Architecture**: YOLOv8 (transfer learning from HuggingFace)
- **Input Size**: 960×960 pixels
- **Optimizer**: AdamW (lr=0.001, cosine annealing)
- **Batch Size**: 8
- **Epochs**: 21 (early stopping at epoch 1 due to high initial performance)
- **Augmentation**: HSV shift, rotation, translation, flip, mosaic

### Training Results
- **Best mAP@0.5**: 63.4% (validation set, epoch 1)
- **Final mAP@0.5**: 84.6% (test set after fine-tuning)
- **Training Time**: 2.3 hours on single GPU (Tesla T4)

---

## 🔍 Key Features

### 1. Custom Annotation
- Annotated using CVAT (Computer Vision Annotation Tool)
- Frame-by-frame labeling from video sequences
- Quality verification and consistency checks

### 2. Real-time Processing
- Video frame processing at competitive inference speeds
- Multi-object tracking for player continuity
- Automated classification of ball possession

### 3. Analytics Pipeline
- Ball possession percentage by team
- Pass detection and tracking
- Team-specific performance metrics
- Referee position tracking

### 4. Robustness
- Handles varying lighting conditions
- Works across different pitch layouts
- Resilient to occlusions and overlapping players

---

## 📁 Files Included

### Notebooks
- **`Before_Training.ipynb`** - Baseline evaluation on original HuggingFace football model
- **`Best_Results.ipynb`** - Complete training, validation, and analysis pipeline

### Data Files
- **CSV Results**: Detailed per-class metrics, error analysis, threshold comparisons
- **Visualizations**: Confusion matrices, training curves, result plots
- **Model**: Trained `best.pt` weights (21.5MB)

### Documentation
- This comprehensive README
- Per-file metadata and descriptions

---

## 🚀 How to Use

### Prerequisites
```bash
pip install ultralytics opencv-python pandas numpy torch torchvision
pip install huggingface_hub pyyaml scikit-learn
```

### Loading the Model
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference
results = model.predict(source='video.mp4', imgsz=960, conf=0.25)

# Access predictions
for result in results:
    boxes = result.boxes  # Object bounding boxes
    classes = result.boxes.cls  # Class IDs (0=player, 1=ball, 2=referee)
    confs = result.boxes.conf  # Confidence scores
```

### Class Mapping
- **Class 0**: Player
- **Class 1**: Ball
- **Class 2**: Referee

---

## 📈 Performance Insights

### Strengths
✅ Excellent player detection (98% recall)
✅ Strong referee identification (89% recall)
✅ Minimal false positive player/referee confusion (0.16%)
✅ Fast inference time (< 13ms per frame)
✅ Robust team differentiation

### Areas for Improvement
⚠️ Ball detection needs refinement (56% recall)
  - Small object size challenges
  - Motion blur in video sequences
  - Consider ensemble methods or specialized ball detector

⚠️ Referee occasional misclassification as player (14.6% at threshold 0.25)
  - Possible due to similar appearance when referee moves
  - Could improve with additional referee-specific training data

---

## 🎓 Learning Outcomes

### Skills Developed
- Computer vision and object detection (YOLO framework)
- Dataset creation and annotation management
- Transfer learning and fine-tuning deep learning models
- Real-time video processing and tracking
- Sports analytics and performance metrics
- Python for ML/AI workflows
- Data analysis and visualization

### Challenges Overcome
- Small object detection in sports contexts
- Handling occlusions and crowded scenes
- Balancing precision vs. recall trade-offs
- GPU optimization for real-time inference
- Robust class differentiation in similar-looking objects

---

## 📚 Technologies & Tools

| Category | Tools |
|----------|-------|
| **Detection** | YOLO v8, Ultralytics |
| **Tracking** | Multi-Object Tracking (MOT) |
| **Languages** | Python, Jupyter |
| **Frameworks** | PyTorch, OpenCV |
| **Annotation** | CVAT |
| **GPU** | NVIDIA Tesla T4 |
| **Analysis** | Pandas, NumPy, Scikit-learn |

---

## 📞 About

**Student**: Raad Shraideh  
**University**: Jordan University of Science and Technology (J.U.S.T)  
**Period**: 4 years of AI Study  
**Focus Areas**: Computer Vision, Deep Learning, Real-time Systems

---

## 📝 License

These projects are part of my academic portfolio. Feel free to explore and learn from them.

---

## 🙏 Acknowledgments

- J.U.S.T faculty for guidance and support
- Ultralytics for the YOLOv8 framework
- Open-source community for tools and datasets
- Jordanian football community for inspiring this project

---

**Last Updated**: June 2026  
**Status**: Complete & Documented

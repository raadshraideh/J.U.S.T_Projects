# Football Detection & Player Tracking Project

## 🎯 Project Overview

This project implements real-time football player and ball detection using YOLOv8 with advanced tracking and analytics for Jordanian Pro League matches.

## 📊 Dataset Information

### Custom Dataset Characteristics
- **Source**: Jordanian Pro League match footage
- **Annotation Tool**: CVAT (Computer Vision Annotation Tool)
- **Train/Val/Test Split**: 4280 / 1462 / 720 images
- **Classes**: 3 classes (Player, Ball, Referee)
- **Original Classes**: 4 (Player_Team1, Player_Team2, Ball, Referee) → converted to 3-class

### Dataset Statistics
- **Training**: 43,242 players + 4,925 referees + 3,542 balls
- **Validation**: 16,137 players + 1,614 referees + 1,414 balls
- **Test**: 6,273 players + 787 referees + 641 balls

## 📈 Model Performance

### Overall Metrics
- **mAP@0.5**: 84.59%
- **mAP@0.5-0.95**: 0.42
- **Precision (mean)**: 88.55%
- **Recall (mean)**: 81.08%

### Per-Class Performance

| Class    | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|----------|-----------|--------|---------|---------------|
| Player   | 93.05%    | 97.21% | 97.24%  | 56.80%        |
| Ball     | 81.31%    | 57.10% | 63.68%  | 25.02%        |
| Referee  | 91.27%    | 88.95% | 92.84%  | 44.23%        |

### Advanced Metrics (Threshold = 0.25)
- **Player Detection Recall**: 98.1%
- **Ball Recall**: 56.16%
- **Referee Recall**: 88.95%
- **Team A Detection**: 98.17%
- **Team B Detection**: 98.04%
- **Referee wrongly detected as Player**: 14.61%
- **Player wrongly detected as Referee**: 0.16%

## 🚀 Getting Started

### Prerequisites
```bash
pip install ultralytics torch torchvision opencv-python pandas numpy scikit-learn pyyaml
```

### Running the Notebooks

1. **Before_Training.ipynb** - Baseline model evaluation on the original 4-class dataset
2. **Best_Results.ipynb** - Complete training pipeline with:
   - Dataset conversion (4-class → 3-class)
   - Weighted sampling for class imbalance
   - Model training with early stopping
   - Test set validation
   - Comprehensive error analysis
   - Threshold comparison analysis

## 📁 Files in This Directory

- `Before_Training.ipynb` - Pre-training evaluation (~70 MB)
- `Best_Results.ipynb` - Complete training & analysis (~50 MB)
- `results/` - Output CSVs, confusion matrices, and plots
- `README.md` - This file

## 🔍 Key Notebooks Breakdown

### Before_Training.ipynb
- Loads pre-trained HuggingFace model
- Converts 4-class labels to 3-class
- Validates on test set without fine-tuning
- Generates baseline metrics and confusion matrices

### Best_Results.ipynb
**Cell 1-3**: Data preparation
- Finds and validates YOLO dataset
- Verifies train/val/test splits
- Converts 4-class → 3-class labels

**Cell 4**: Weighted training list creation
- Oversamples images containing ball and referee
- Creates balanced training distribution

**Cell 5**: Model training
- Fine-tunes YOLOv8 from HuggingFace model
- 80 epochs with AdamW optimizer
- Early stopping (patience=20)
- Saves checkpoints for robustness

**Cell 6**: Test set validation
- Comprehensive metrics on test set
- Per-class performance analysis
- Confusion matrices

**Cell 7**: Error analysis
- Greedy matching with IOU threshold (0.5)
- Team-based statistics
- Misclassification analysis
- Confusion between classes

**Cell 8**: Threshold comparison
- Tests multiple confidence thresholds (0.1-0.4)
- Measures recall/precision trade-offs
- Identifies optimal threshold for deployment

**Cell 9**: Final export
- Packages all results into ZIP
- Includes model, CSVs, and visualizations

## 📊 Output Files

### CSV Files
- `three_class_swapped_summary.csv` - Overall training metrics
- `three_class_swapped_per_class.csv` - Per-class performance
- `swapped_error_analysis_summary.csv` - Error analysis across test set
- `swapped_image_error_analysis.csv` - Per-image error breakdown
- `swapped_threshold_comparison.csv` - Threshold analysis results

### Visualization Files
- `confusion_matrix.png` - Raw confusion matrix
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `results.png` - Training curves (loss, mAP, etc.)

## 🎓 Technical Highlights

### Data Processing
- Automated dataset discovery and validation
- Class-aware weighted sampling
- IOU-based greedy matching for evaluation
- Robust error handling for Kaggle session restarts

### Model Architecture
- Base: YOLOv8n (nano) pretrained on COCO
- Transfer learning from HuggingFace football detection model
- Fine-tuned on custom Jordanian Pro League dataset
- ~11.1M parameters with 28.4 GFLOPs

### Training Strategy
- Mixed precision (AMP) for faster training
- Cosine learning rate schedule
- Heavy augmentation (mosaic, mixup, HSV variations)
- Early stopping to prevent overfitting
- Save period = 1 epoch for checkpoint safety

## 💡 Key Insights

1. **High Player Detection**: 98%+ recall on both teams indicates reliable player tracking capability
2. **Ball Detection Challenge**: 56% ball recall reflects difficulty in small object detection
3. **Team Separation**: Excellent at differentiating Team A vs Team B (98%+ each)
4. **Referee Distinction**: 89% accuracy in identifying referees, with only 14.6% false positives as players
5. **Threshold Sensitivity**: Ball recall is most sensitive to confidence threshold; player recall remains stable

## 🔧 Troubleshooting

### Kaggle Session Restart
The notebooks are designed to handle Kaggle session restarts:
- Models are saved every epoch
- Results are exported at each stage
- Helper functions locate models robustly
- If session restarts, rerun from top of notebook

### Memory Issues
- Reduce batch size from 8 to 4 in Cell 5
- Reduce image size from 960 to 640
- Use `cache=False` to avoid cache files

## 📚 References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- HuggingFace Model: uisikdag/yolo-v8-football-players-detection
- CVAT Annotation Tool: https://www.cvat.ai/

## 📞 Project Contact

**Author**: Raad Shraideh  
**University**: Jordan University of Science and Technology (J.U.S.T)  
**Year**: 2024-2026  

---

**Note**: This project demonstrates production-ready ML practices including robust error handling, checkpoint management, and comprehensive evaluation.

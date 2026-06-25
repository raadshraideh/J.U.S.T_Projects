# J.U.S.T Projects Portfolio

A collection of AI/ML projects developed during my 4 years of study at **Jordan University of Science and Technology (J.U.S.T)**.

## 📋 Projects

### 1. Football Detection & Player Tracking
**Status**: ✅ Complete  
**Language**: Python (Jupyter Notebooks)  
**Technologies**: YOLOv8, OpenCV, PyTorch, Object Detection

Built a custom-labeled Computer Vision dataset for the Jordanian Pro League with:
- Real-time YOLO and MOT (Multi-Object Tracking) deployment pipeline
- Automated classification logic for ball possession analysis
- Pass tracking and team-based metrics separation (Team A/Team B)
- **Achieved 84.6% mAP@0.5** on custom dataset with 3-class detection (Player, Ball, Referee)

**Directory**: `./01_Football_Detection/`

---

## 📁 Directory Structure

```
J.U.S.T_Projects/
├── README.md                          # This file
├── 01_Football_Detection/
│   ├── README.md                      # Project-specific documentation
│   ├── Before_Training.ipynb          # Model evaluation before fine-tuning
│   ├── Best_Results.ipynb             # Model after training on custom dataset
│   ├── results/                       # Training outputs, logs, and analysis
│   └── data/                          # Dataset information and preprocessing
├── 02_[Future_Project]/
└── ...
```

## 🎯 Key Achievements

- **Custom Dataset**: Annotated Jordanian Pro League match footage using CVAT tool
- **Model Performance**: 84.6% mAP@0.5 with 98.1% player recall
- **Real-time Detection**: Deployed inference pipeline for live match analysis
- **Advanced Metrics**: Implemented ball possession, pass tracking, and team separation

## 🛠️ Tech Stack

- **ML/DL**: PyTorch, YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Annotation**: CVAT
- **Deployment**: Kaggle Notebooks

## 📚 How to Use

1. Navigate to the specific project folder
2. Read the project's `README.md` for detailed instructions
3. Run the Jupyter notebooks in the specified order
4. Check `results/` for outputs, logs, and analysis

## 📝 Notes

Each project folder is self-contained with its own documentation. New projects will be added following this structure.

---

**Last Updated**: June 2026  
**Author**: [Your Name]  
**University**: Jordan University of Science and Technology (J.U.S.T)

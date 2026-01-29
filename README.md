# Veterinary Disease Classification in Goats Using CRNNs

## Project Overview

This project focuses on the automated classification of goat health status for veterinary disease detection using image-based analysis. The system classifies goats into Healthy and Unhealthy categories using a Convolutional Recurrent Neural Network (CRNN) implemented entirely from scratch, without using any pretrained models.

## Key Features

- CRNN architecture combining CNN feature extraction with GRU sequential learning
- From-scratch implementation without pretrained models
- Test accuracy of 94.44%
- Comprehensive data preprocessing and augmentation
- Model evaluation with confusion matrix analysis

## Team Members

- Shahan Anwar (Registration: su92-msaiw-f25-006)
- Bisma Ashraf (Registration: su92-msaiw-f25-016)

## Dataset

The dataset consists of goat images organized into two categories:
- healthy_goat: Images of healthy goats
- unhealthy_goat: Images of goats showing signs of disease

Images are preprocessed to 128x128 pixels and split into training, validation, and test sets.

## Model Architecture

- **CNN Feature Extractor**: 4 convolutional layers (16, 32, 64, 128 filters) with batch normalization and max pooling
- **Reshape Layer**: Converts feature maps to sequences
- **GRU Layer**: 128-unit Gated Recurrent Unit for sequential learning
- **Classifier**: Dense layers with dropout and softmax activation

Total Parameters: 558,306 (2.13 MB)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd GOAT_AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your dataset in the following structure:
```
Goat_Splits/
├── train/
│   ├── healthy_goat/
│   └── unhealthy_goat/
├── val/
│   ├── healthy_goat/
│   └── unhealthy_goat/
└── test/
    ├── healthy_goat/
    └── unhealthy_goat/
```

## Usage

1. Update the BASE_DIR path in `goat_crnn.py` to point to your dataset
2. Run the training script:
```bash
python goat_crnn.py
```

The script will:
- Load and preprocess the dataset
- Train the CRNN model
- Evaluate on test set
- Generate confusion matrix
- Save the trained model

## Results

- **Test Accuracy**: 94.44%
- **Training Accuracy**: 95.03%
- **Validation Accuracy**: 92.88%

## Project Structure

```
GOAT_AI/
├── goat_crnn.py              # Main training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── project_documentation.md  # Comprehensive documentation
├── final_report.txt         # Final project report
├── team_introduction.txt    # Team member introductions
├── github_commit_history.txt # Git commit history
├── demo_video_script.txt    # Demo video script
├── results.txt              # Training results and metrics
└── [model files and outputs]
```

## Framework and Tools

- **Framework**: TensorFlow 2.x / Keras
- **Language**: Python 3.x
- **Platform**: Google Colab
- **Visualization**: Matplotlib, Seaborn

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Loss (Categorical Cross-entropy)

## License

This project is developed for academic purposes as part of MS-AI coursework.

## Acknowledgments

Developed as part of MS-AI coursework requirements. Uses Google Colab for computational resources and TensorFlow/Keras framework for model development.


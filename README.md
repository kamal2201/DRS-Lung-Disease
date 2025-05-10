
# 🦠 COVID-19 Detection System Using CNN

This project is a Deep Learning-based web application for detecting COVID-19 from chest X-ray images using a Convolutional Neural Network (CNN). It includes both a model training notebook and a Flask-based web interface for real-time inference.

## 📁 Project Structure

```
.
├── app.py
├── building-a-covid-19-detection-system-using-cnn-dl.ipynb
├── models/
│   ├── CNN_Covid19_Xray_Version.h5
│   └── Label_encoder.pkl
├── templates/
│   └── index.html
└── static/
    └── uploaded/
```

## 🧠 Model Overview

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: Chest X-ray images (grayscale/RGB)
- **Output Classes**: COVID-19 / Normal / Pneumonia (based on Label Encoder)
- **Training Workflow**: Data preprocessing → CNN model training → Evaluation → Save model & encoder

## 🌐 Web Application Features

- Upload X-ray image
- Display prediction with label
- Uses pretrained CNN model
- Google Maps API integration (not central to prediction flow)

## 🚀 How to Run

### Requirements

- Python 3.7+
- Jupyter Notebook (for training)
- Flask (for serving)
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

### Setup Instructions

```bash
git clone https://github.com/your-repo/covid19-xray-detector.git
cd covid19-xray-detector
pip install -r requirements.txt
```

### Training the Model

Launch Jupyter Notebook and run:

```bash
jupyter notebook building-a-covid-19-detection-system-using-cnn-dl.ipynb
```

Ensure the trained model (`CNN_Covid19_Xray_Version.h5`) and label encoder (`Label_encoder.pkl`) are stored in the `models/` directory.

### Running the Web App

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## ⚠️ Important

- Do not use this model for clinical decisions.
- Hide your Google Maps API key in production.

## 📊 Results

Training and validation accuracy and loss graphs can be found in the notebook.

## 🙌 Acknowledgements

- Kaggle COVID-19 X-ray Dataset
- TensorFlow/Keras
- Flask Framework

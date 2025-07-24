# 👁️‍🗨️ Employee Recognition System using EfficientNet

In today’s workplace, secure and seamless access control is essential for both employee experience and organizational safety. This project presents an AI-powered facial recognition system using an EfficientNet-based model trained on employee faces to identify authorized personnel and distinguish intruders.

Imagine employees walking into the building and being greeted not by a security guard, but by an intelligent system that recognizes their face and grants access instantly, **no badges, no delays, no bottlenecks**. This project is a step toward **secure, automated, and intelligent access management**.

---

## 📌 Project Objectives

- Train a facial recognition model to accurately identify employees
- Evaluate model performance on real-world testing scenarios
- Integrate the recognition model into a mock access control pipeline
- Enable potential real-world deployment in a security system

---

## 🛠️ Main Tasks & Tools Used

| Task                          | Description                                      | Tools / Libraries              |
|-------------------------------|--------------------------------------------------|--------------------------------|
| Data Preprocessing            | Loading and preparing face datasets              | `Python`, `OpenCV`, `NumPy`    |
| Model Architecture            | Building EfficientNet-based classifier           | `TensorFlow`, `Keras`, `EfficientNet` |
| Training Pipeline             | Training the model on original & augmented data  | `Keras`, `TensorFlow`, `Pandas` |
| Model Evaluation              | Assessing accuracy, loss, and recognition scores | `matplotlib`, `sklearn`, `NumPy` |
| Recognition System Integration| Real-time face recognition testing               | `EfficientNet`, `OpenCV`       |

---

## 📂 Project Structure

```
├── Data_processing/            # Data preparation scripts
│   ├── __init__.py
│   └── data_loader.py          # Dataset loading utilities
│
├── Model_Development/          # Model training code
│   ├── __init__.py
│   ├── model_builder.py        # Model architecture
│   ├── predict.py              # Inference functions
│   └── train_model.py          # Training pipeline
│
├── Model_Evaluation/           # Performance metrics
│   ├── __init__.py
│   ├── metrics_training.py     # Training metrics
│   └── metrics_validation.py   # Validation metrics
│
├── Models/                     # Saved model files
│   ├── model_augmented.keras   # Augmented data model
│   └── model_original.keras    # Original data model
│
├── Recognition_System/         # Production system
│   ├── __init__.py
│   ├── efficientnet_program.py # Main recognition logic
│   └── image_testing.py        # Test scripts
│
├── dataset/                    # Employee face dataset
├── main.py                     # Application entry point
└── requirements.txt            # Python dependencies
```

---

## 📈 Model Overview

- **Model Type**: Image classifier using **EfficientNet-B0**
- **Input**: Face images (preprocessed to fixed dimensions)
- **Output**: Employee identity class (or unknown)
- **Versions**:
  - `model_original.keras`: Trained with original dataset
  - `model_augmented.keras`: Trained with data augmentation (improved generalization)

---

## 🧪 How to Run

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/employee-recognition.git
   cd employee-recognition
   ```

2. **Set up the environment**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional if not using pretrained)
   ```bash
   python Model_Development/train_model.py
   ```

4. **Run the recognition system**
   ```bash
   python Recognition_System/efficientnet_program.py
   ```

5. **Test with sample images**
   ```bash
   python Recognition_System/image_testing.py
   ```

---

## 📊 Evaluation Metrics

- **Training Accuracy**
- **Validation Accuracy**
- **Loss Curves**
- **Confusion Matrix**
- **Recognition Time per Image (ms)**

---

## 🛡️ Real-World Application

🔒 **Integration with security systems**: The recognition script can be plugged into an access control device (e.g. Raspberry Pi + camera) to allow or deny physical entry.

---

## 🧑‍💻 Author

Chris Essomba – *Deep Learning Engineer & Computer Vision Enthusiast*

---

## 📄 License

MIT License — Feel free to use, modify, and distribute with proper attribution.


# Emotion Detection from Behind-the-Ear PPG Signals

This project implements deep learning models for **emotion recognition using PPG signals collected from a behind-the-ear wearable device**.  
It is based on our published paper:

📄 Fatima Zahra *et al.*, **"Emotion Recognition from Behind-the-Ear PPG Signals Using Deep Learning"**, [Journal/Conference name, Year].  

---

## 📂 Repository Structure
- `preprocess.py` → Filtering (Butterworth), normalization.  
- `model.py` → 1D-CNN architecture for emotion classification.  
- `train.py` → Training script (requires dataset access).  
- `predict.py` → Inference on test PPG signals.  
- `requirements.txt` → Python dependencies.  

---

## 🚀 Usage
### 1. Train
```bash
python train.py
```
(Add your dataset loader in `train.py`.)

### 2. Predict
```bash
python predict.py --weights ppg_emotion_model.pth
```

---

## 📊 Dataset Access
The dataset used in this work is **not publicly available**.  
To request access, please **cite our paper**:

> Fatima Zahra *et al.*, **"Emotion Recognition from Behind-the-Ear PPG Signals Using Deep Learning"**, [Journal/Conference name, Year].

Then contact the authors with proof of citation.

---

## 🧠 Techniques Used
- Preprocessing: Butterworth bandpass filtering, normalization.  
- Model: 1D-CNN for temporal features (5 emotions: happy, sad, angry, calm, fear).  
- Evaluation: Accuracy, F1-score, Confusion Matrix.  

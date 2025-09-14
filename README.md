
# Emotion Detection from Behind-the-Ear PPG Signals

This project implements deep learning models for **emotion recognition using PPG signals collected from a behind-the-ear wearable device**.  
It is based on our published paper:

📄 Emotion Recognition from Behind-the-Ear Photoplethysmography Signal Using Continuous Wavelet Transform and Deep Learning
Journal of the Society for Convergence Signal Processing
Abbreviations: JISPS
2025 vol.26, no.1, pp.1 - 9
DOI : 10.23087/jkicsp.2025.26.1.001
Publisher : Korean Society for Convergence Signal Processing

---

## 📂 Repository Structure
- `preprocess.py` → Filtering (Butterworth), normalization.  
- `model.py` → CNN architecture for emotion classification.  
- `train.py` → Training script (requires dataset access).  
- `predict.py` → Inference on test PPG signals.  
- `requirements.txt` → Python dependencies.  

---

## 🚀 Usage
### 1. Train
```bash
python train.py
```


### 2. Predict
```bash
python predict.py --weights ppg_emotion_model.pth
```

---

## 📊 Dataset Access
The dataset used in this work is **not publicly available**.  
To request access, please **cite our paper**:

>Emotion Recognition from Behind-the-Ear Photoplethysmography Signal Using Continuous Wavelet Transform and Deep Learning.
>2025 vol.26, no.1, pp.1 - 9
DOI : 10.23087/jkicsp.2025.26.1.001
Publisher : Korean Society for Convergence Signal Processing
> Includes preprocessing, training, and inference scripts. Dataset not public; please cite our paper to request access.

Then contact the authors with proof of citation.

---

##  Techniques Used
- Preprocessing: Butterworth bandpass filtering, normalization.  
- Model: CNN for temporal features (5 emotions: happy, sad, angry, calm, fear).  
- Evaluation: Accuracy, F1-score, Confusion Matrix.

  Preprocessing Of PPG:P
  <img width="860" height="371" alt="image" src="https://github.com/user-attachments/assets/d44e3bd4-8327-4b94-93b7-26e976b25b37" />


  Rseults:
  <img width="1676" height="676" alt="image" src="https://github.com/user-attachments/assets/75ff27bf-0177-4a01-b444-31819927bf2e" />
<img width="1507" height="626" alt="image" src="https://github.com/user-attachments/assets/b5d7e6df-4361-4b2d-ad2c-3f096c943d61" />

   

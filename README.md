# Deepfake Audio Detection

This project implements a **Deepfake Audio Detection** pipeline using **deep learning models (VGG16 and LSTM)** to classify **real vs. AI-generated (fake)** voice samples.
The notebook automates data preprocessing, feature extraction, model training, evaluation, and visualization.

---

## 📘 Overview

With the rise of AI-generated synthetic voices, distinguishing real from fake audio has become crucial for security and authenticity verification.
This project aims to **detect deepfake audio** using **spectrogram-based features** and **deep learning architectures** for robust classification.

The notebook handles:

* Audio data loading and deduplication
* MFCC spectrogram extraction
* Deep learning model creation (VGG16 and LSTM)
* Model training, evaluation, and visualization

---

## 🧩 Workflow

### 1. **Data Loading**

The dataset is downloaded from Kaggle (`mohammedabdeldayem/the-fake-or-real-dataset`) or loaded locally.
It contains two folders:

```
real/
fake/
```

Each folder contains `.wav` files of real or synthetic audio clips.

The code:

* Loads all `.wav` files
* Removes duplicates using MD5 hashing
* Pads shorter audios to a fixed length (1 second, 16 kHz)

---

### 2. **Feature Extraction**

Each audio file is converted into **MFCC (Mel-Frequency Cepstral Coefficients)** spectrograms:

* 40 MFCCs per frame
* Normalized to zero mean and unit variance
* Padded or trimmed to a fixed time dimension (40×64)
* Expanded to a 4D tensor for CNN input

```python
mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
```

---

### 3. **Model Architectures**

#### 🧠 VGG16 Model

A **transfer learning** model using pretrained **VGG16** (ImageNet weights):

* Converts MFCC to 3-channel format
* Freezes base layers of VGG16
* Adds dense layers for binary classification

```python
x = Flatten()(vgg_output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)
```

#### 🔁 LSTM Model

An **LSTM network** analyzes sequential MFCC features:

* Two stacked LSTM layers
* Dropout regularization
* Dense layers for final classification

```python
LSTM(128, return_sequences=True)
LSTM(64)
Dense(32, activation='relu')
Dense(2, activation='softmax')
```

---

### 4. **Training and Evaluation**

* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam
* **Metrics:** Accuracy

Models are trained with:

* Early stopping (`patience=5`)
* Checkpointing for best model saving

Both models are evaluated on the test set with:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

Visual outputs include:

* Accuracy plots per epoch
* Confusion matrices for real vs. fake predictions

---

### 5. **Results and Visualization**

#### 📊 Accuracy and Confusion Matrix

* Side-by-side comparison of VGG16 and LSTM performance
* Accuracy and validation trends across epochs
* Confusion matrix visualizations saved as `.png` files

#### 🧾 Metrics Summary

Classification reports are printed for both models, including:

* Class-wise precision, recall, F1-score
* Weighted averages

---

## 🛠️ Dependencies

Install the required Python packages before running:

```bash
pip install tensorflow librosa numpy pandas matplotlib scikit-learn tqdm kagglehub
```

---

## 📂 File Outputs

* `vgg16_full.h5` — Saved trained VGG16 model
* `lstm_full.h5` — Saved trained LSTM model
* `{dataset_name}_training_history.png` — Training curves
* `{dataset_name}_confusion_matrices.png` — Confusion matrices

---

## ⚙️ Execution

1. Update dataset path in the notebook:

   ```python
   dataset_dirs = {
       "norm": "path/to/for-norm"
   }
   ```
2. Run all cells sequentially.
3. View metrics and visualizations in the output cells.

---

## 📈 Summary

| Model | Accuracy                             | Precision                                | Recall | F1-score |
| ----- | ------------------------------------ | ---------------------------------------- | ------ | -------- |
| VGG16 | High visual feature extraction power | Excellent for spectrogram classification |        |          |
| LSTM  | Captures temporal dependencies       | Complements CNN results                  |        |          |


## 🧠 Keywords

`Deepfake Detection` • `Audio Classification` • `MFCC` • `VGG16` • `LSTM` • `TensorFlow` • `Librosa`

---

This repository provides a solid baseline for **AI-generated voice detection** using deep learning on spectrogram features.

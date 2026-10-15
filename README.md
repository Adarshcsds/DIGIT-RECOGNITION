# Handwritten Digit Recognition

An end-to-end handwritten digit recognition project built with **TensorFlow/Keras** and **Streamlit**. It compares a Perceptron, a fully connected Artificial Neural Network (ANN), and a Convolutional Neural Network (CNN), then lets users upload an image and inspect each model's prediction.

<p align="center">
  <a href="https://digit-recognition-aufcnfsmnovappvocmqvhxz.streamlit.app/"><strong>🚀 Try the Live Demo</strong></a>
  &nbsp;•&nbsp;
  <a href="https://www.kaggle.com/competitions/digit-recognizer"><strong>Kaggle Digit Recognizer</strong></a>
</p>



## Highlights

- Interactive Streamlit app for handwritten digit images (0–9).
- Image preprocessing: grayscale conversion, foreground detection/cropping, padding, resizing, and centering on a 28 × 28 canvas.
- Side-by-side predictions from three trained neural-network approaches.
- Class probability/confidence visualization.
- Kaggle submission generated from the CNN model.

> **Note:** A model's displayed confidence is its predicted class probability; it is not a guarantee that the prediction is correct.

## Results

### Model comparison

The following scores were calculated on an 8,400-image validation split from the Kaggle training dataset.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|---|---:|---:|---:|---:|
| Perceptron | 89.15% | 89.03% | 88.98% | 88.98% |
| ANN | 96.35% | 96.36% | 96.34% | 96.33% |
| **CNN** | **98.81%** | **98.80%** | **98.80%** | **98.80%** |

The CNN was the strongest of the three models on this validation split.

### Kaggle competition

- **Kaggle score:** **0.98764** (98.764%).
- **Submission file:** `submission.csv`, containing 28,000 predicted labels.

The Kaggle score is from the competition's hidden test labels. The local validation metrics are reported separately and should not be confused with the Kaggle score.

## Models

| Model | Architecture / role |
|---|---|
| Perceptron | Flattened 28 × 28 pixels followed by a 10-class softmax output; baseline |
| ANN | Fully connected network with 128- and 64-unit ReLU layers and a 10-class softmax output |
| CNN | Convolution and max-pooling layers, followed by a dense layer, dropout, and a 10-class softmax output |

All three models are saved in Keras format.

## Tech stack

- **Language:** Python
- **Deep learning:** TensorFlow, Keras
- **Data processing:** NumPy, pandas
- **Image processing:** OpenCV, Pillow
- **Evaluation:** scikit-learn, Matplotlib, Seaborn
- **Application:** Streamlit
- **Dataset / competition:** Kaggle Digit Recognizer

## Repository structure

```text
DIGIT-RECOGNITION/
├── app.py                 # Streamlit application
├── CNN.ipynb              # Data preparation, training, and evaluation notebook
├── perceptron.keras       # Saved baseline model
├── ann.keras              # Saved ANN model
├── digit_cnn.keras        # Saved CNN model
├── requirements.txt       # Python dependencies
├── sample_submission.csv  # Kaggle submission template
└── .gitignore
```

The Kaggle `train.csv` and `test.csv` datasets are kept out of the repository to avoid committing large data files. Download them from the competition page when reproducing notebook experiments.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Adarshcsds/DIGIT-RECOGNITION.git
cd DIGIT-RECOGNITION
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

Ensure the three `.keras` files are in the project root. To run the training/evaluation notebook, download the Kaggle data and place `train.csv` at `train/train.csv` and `test.csv` at `test/test.csv`.

## How to use the app

1. Open the [live demo](https://digit-recognition-aufcnfsmnovappvocmqvhxz.streamlit.app/) or run it locally.
2. Upload a clear image containing one handwritten digit.
3. Review the processed image, predicted digit, and confidence from each model.
4. Compare the models' outputs.

For best results, use an image with one clearly visible digit and enough contrast against its background.

## Future improvements

- Add automated tests for preprocessing and model loading.
- Add version-pinned dependencies for more reproducible environments.
- Improve robustness to unusual handwriting, lighting, and image backgrounds.

## Author

**Adarsh Dubey** · [GitHub profile](https://github.com/Adarshcsds)

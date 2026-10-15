# Handwritten Digit Recognition

An interactive machine-learning application that predicts handwritten digits (0–9) from uploaded images and compares three trained models: Perceptron, Artificial Neural Network (ANN), and Convolutional Neural Network (CNN).

## Live Demo

**[Open the Streamlit app](https://digit-recognition-aufcnfsmnovappvocmqvhxz.streamlit.app/)**

## Features

- Upload a PNG, JPG, or JPEG image containing one handwritten digit.
- Preprocess the image into a centered 28 × 28 grayscale representation.
- Compare predictions from Perceptron, ANN, and CNN models.
- Display each model's predicted digit and confidence score.
- Visualize confidence and class-probability comparisons.

> Confidence is the model's predicted probability for its selected class; it is not the same as measured accuracy.

## Models

| Model | Description |
|---|---|
| Perceptron | Baseline model for comparison |
| ANN | Fully connected neural network |
| CNN | Convolutional neural network for image classification |

Model performance should be compared using the same held-out labeled dataset. No accuracy figures are listed here until they have been measured and verified.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- Streamlit

## Project Structure

```text
DIGIT-RECOGNITION/
├── app.py
├── CNN.ipynb
├── perceptron.keras
├── ann.keras
├── digit_cnn.keras
├── requirements.txt
└── .gitignore
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Adarshcsds/DIGIT-RECOGNITION.git
   cd DIGIT-RECOGNITION
   ```

2. Create and activate a virtual environment.

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

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

The three `.keras` model files must be present in the project root for the app to load them.

## Dataset

The models were developed for 28 × 28 handwritten digit classification using the Kaggle Digit Recognizer dataset.

Dataset: [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)

The dataset files are not required to run the deployed inference app if the trained model files are included. Do not commit the full dataset unless there is a clear reason to do so.

## Future Improvements

- Evaluate all models on the same held-out test split and report accuracy.
- Add a confusion matrix and per-class metrics.
- Add automated tests for image preprocessing and model input shapes.
- Pin compatible dependency versions for reproducible deployment.
- Include a screenshot or GIF of the application.

## Author

**Adarsh Dubey** — [GitHub profile](https://github.com/Adarshcsds)


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image


st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Digit Recognition")
st.write("Upload an image containing one digit from 0 to 9.")



@st.cache_resource
def load_models():
    return {
        "Perceptron": tf.keras.models.load_model("perceptron.keras"),
        "ANN": tf.keras.models.load_model("ann.keras"),
        "CNN": tf.keras.models.load_model("digit_cnn.keras"),
    }




def preprocess_image(image):
    # Convert to grayscale
    gray = np.array(image.convert("L"))

    # Blur slightly to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Estimate background brightness from the borders
    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])
    background_is_light = np.median(border) > 127

    # Threshold according to background polarity
    if background_is_light:
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Crop foreground while preserving all foreground pixels
    coords = cv2.findNonZero(binary)
    if coords is None:
        raise ValueError("No digit detected. Try a clearer image.")

    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y+h, x:x+w]

    # Add padding before resizing to preserve shape
    pad = max(2, int(max(w, h) * 0.15))
    digit = cv2.copyMakeBorder(
        digit, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=0
    )

    # Resize proportionally
    digit_image = Image.fromarray(digit)
    digit_image.thumbnail((20, 20))

    # Center on the black background used by MNIST-style models
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(
        digit_image,
        ((28 - digit_image.width) // 2,
         (28 - digit_image.height) // 2)
    )

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    return arr, canvas




uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Original uploaded image")

    
if st.button("Compare All Models", type="primary"):
    try:
        models = load_models()

        # Keep your existing preprocessing function
        processed, preview = preprocess_image(image)

        st.subheader("Processed image")
        st.image(preview, width=180)

        # Perceptron and ANN were trained on (28, 28) images.
        # CNN was trained on (28, 28, 1) images.
        flat_input = processed.reshape(1, 28, 28)

        inputs = {
            "Perceptron": flat_input,
            "ANN": flat_input,
            "CNN": processed
        }

        results = []

        for name, model in models.items():
            probabilities = model.predict(
                inputs[name],
                verbose=0
            )[0]

            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities)) * 100

            results.append({
                "Model": name,
                "Predicted Digit": predicted_digit,
                "Confidence (%)": round(confidence, 2)
            })

        # Display model comparison
        st.subheader("Model Prediction Comparison")

        col1, col2, col3 = st.columns(3)

        for col, result in zip(
            [col1, col2, col3], results
        ):
            with col:
                st.metric(
                    label=result["Model"],
                    value=f"Digit {result['Predicted Digit']}",
                    delta=f"{result['Confidence (%)']:.2f}% confidence"
                )

        # Tabular comparison
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # Compare confidence for this uploaded image
        st.subheader("Confidence Comparison")

        confidence_chart = {
            result["Model"]: result["Confidence (%)"]
            for result in results
        }

        st.bar_chart(confidence_chart, y_label="Confidence (%)")

        # Show all ten class probabilities per model
        st.subheader("Probability Breakdown")

        probability_data = {}

        for name, model in models.items():
            probabilities = model.predict(
                inputs[name],
                verbose=0
            )[0]

            probability_data[name] = probabilities * 100

        st.line_chart({
            "Digit": list(range(10)),
            **{
                name: probabilities
                for name, probabilities in probability_data.items()
            }
        } if False else {
            str(digit): {
                name: float(probabilities[digit])
                for name, probabilities in probability_data.items()
            }
            for digit in range(10)
        }, x_label="Digit", y_label="Probability (%)")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

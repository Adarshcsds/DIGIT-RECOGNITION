import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered",
)

st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload an image containing one handwritten digit from 0 to 9 "
    "to compare three trained models."
)


@st.cache_resource
def load_models():
    """Load the saved models once and reuse them across reruns."""
    return {
        "Perceptron": tf.keras.models.load_model("perceptron.keras"),
        "ANN": tf.keras.models.load_model("ann.keras"),
        "CNN": tf.keras.models.load_model("digit_cnn.keras"),
    }


def preprocess_image(image):
    """Convert an uploaded image to a centered 28x28 digit image."""
    gray = np.array(image.convert("L"))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )
    background_is_light = np.median(border) > 127

    threshold_type = (
        cv2.THRESH_BINARY_INV if background_is_light else cv2.THRESH_BINARY
    )
    _, binary = cv2.threshold(
        gray, 0, 255, threshold_type + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)
    if coords is None:
        raise ValueError("No digit detected. Try a clearer image.")

    x, y, width, height = cv2.boundingRect(coords)
    digit = binary[y : y + height, x : x + width]

    padding = max(2, int(max(width, height) * 0.15))
    digit = cv2.copyMakeBorder(
        digit,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    digit_image = Image.fromarray(digit)
    digit_image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(
        digit_image,
        (
            (28 - digit_image.width) // 2,
            (28 - digit_image.height) // 2,
        ),
    )

    array = np.array(canvas, dtype=np.float32) / 255.0
    return array.reshape(1, 28, 28, 1), canvas


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original uploaded image", use_container_width=True)

if st.button(
    "Compare All Models",
    type="primary",
    disabled=uploaded_file is None,
):
    try:
        models = load_models()
        processed, preview = preprocess_image(image)

        st.subheader("Processed image")
        st.image(preview, caption="28 × 28 model input", width=180)

        # These input shapes must match how the models were trained.
        flat_input = processed.reshape(1, 28, 28)
        model_inputs = {
            "Perceptron": flat_input,
            "ANN": flat_input,
            "CNN": processed,
        }

        results = []
        probability_data = {}

        for name, model in models.items():
            probabilities = model.predict(
                model_inputs[name], verbose=0
            )[0]

            # This confidence calculation assumes the model returns
            # class probabilities, typically from a softmax output.
            predicted_digit = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities)) * 100

            results.append(
                {
                    "Model": name,
                    "Predicted Digit": predicted_digit,
                    "Confidence (%)": round(confidence, 2),
                }
            )
            probability_data[name] = probabilities * 100

        st.subheader("Model Prediction Comparison")
        columns = st.columns(len(results))

        for column, result in zip(columns, results):
            with column:
                st.metric(
                    label=result["Model"],
                    value=f"Digit {result['Predicted Digit']}",
                    delta=f"{result['Confidence (%)']:.2f}% confidence",
                )

        st.dataframe(results, use_container_width=True, hide_index=True)

        st.caption(
            "Confidence is the model's score for its selected class; "
            "it is not the same as measured test accuracy."
        )

        st.subheader("Confidence Comparison")
        confidence_chart = {
            result["Model"]: result["Confidence (%)"]
            for result in results
        }
        st.bar_chart(confidence_chart, y_label="Confidence (%)")

        st.subheader("Probability for Each Digit")
        probability_table = {
            "Digit": list(range(10)),
            **{
                name: [float(value) for value in values]
                for name, values in probability_data.items()
            },
        }
        st.line_chart(
            probability_table,
            x="Digit",
            y=list(probability_data.keys()),
            y_label="Probability (%)",
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that all three .keras model files are present and "
            "that each model's expected input shape matches app.py."
        )

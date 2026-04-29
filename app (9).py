
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Define image dimensions (must match training)
IMG_HEIGHT = 250
IMG_WIDTH = 250

# Load the trained model
# Use st.cache_resource to cache the model loading to improve performance
@st.cache_resource
def load_my_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        # It's good practice to run a dummy prediction to compile the model
        # and avoid lazy loading issues on the first real prediction.
        dummy_input = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3))
        model.predict(dummy_input)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Path to the saved model file
# For local deployment, ensure 'image_classifier_model.h5' is in the same directory as app.py
# If deploying to Streamlit Cloud, you might need to upload this file to your GitHub repo
# or use a cloud storage solution and adjust the path accordingly.
model_file_name = 'image_classifier_model.h5'

# Check if the model file exists locally (for local development/testing)
if not os.path.exists(model_file_name):
    st.error(f"Model file '{model_file_name}' not found. Please ensure it's in the same directory as app.py or update the path.")
    loaded_model = None
elif os.path.getsize(model_file_name) == 0: # Check if file is empty
    st.error(f"Model file '{model_file_name}' is empty. There might have been an error during saving.")
    loaded_model = None
else:
    loaded_model = load_my_model(model_file_name)


# Class names (must match the order used during training)
# Based on train_generator.class_indices: {'keyboard': 0, 'mouse': 1}
class_names = ['keyboard', 'mouse']

st.set_page_config(page_title="Keyboard or Mouse Classifier", page_icon=":camera:")

st.title("Image Classifier: Keyboard or Mouse?")
st.write("Upload an image to classify it as either a keyboard or a mouse using a trained Convolutional Neural Network.")

if loaded_model is None:
    st.warning("Model could not be loaded. Please check the model file path and integrity.")
    st.stop() # Stop the app if model isn't loaded

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for prediction
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)

        # Handle grayscale images by converting to 3 channels
        if img_array.ndim == 2: # Grayscale image (H, W)
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] == 1: # Grayscale with channel (H, W, 1)
            img_array = np.repeat(img_array, 3, axis=-1)
        
        # Ensure image has 3 channels if it somehow ended up with more or less after previous steps
        if img_array.shape[-1] != 3:
            st.error("Image does not have 3 channels after preprocessing. Please check image format.")
            st.stop()

        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)

        # Make prediction
        prediction = loaded_model.predict(img_array)
        score = prediction[0][0]

        # Interpret the prediction
        if score > 0.5:
            predicted_class_index = 1 # mouse
            confidence = score
        else:
            predicted_class_index = 0 # keyboard
            confidence = 1 - score
        
        predicted_class_name = class_names[predicted_class_index]

        st.success(f"Prediction: **{predicted_class_name.capitalize()}**")
        st.write(f"Confidence: {confidence*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image or making prediction: {e}")

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# Load your model
model = load_model('D:/luminar/cnn project/cnn_model.h5')

# Define a function to make predictions
def predict_image(img):
    img_resized = cv2.resize(np.array(img), (150, 150)) # Resize image to match model's expected input size
    img_resized = img_resized.reshape(1, 150, 150, 1)  # Reshape for model input
    predictions = model.predict(img_resized)
    ind = predictions.argmax(axis=1)
    return "PNEUMONIA" if ind.item() == 1 else "NORMAL"

# Streamlit app
st.title("Chest X-Ray Classification")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write("""
This project is a machine learning application designed to classify chest X-ray images as either NORMAL or PNEUMONIA. 
The model used in this application is a Convolutional Neural Network (CNN) trained on a dataset of chest X-ray images. 
The goal is to assist in the early detection of pneumonia, which can be critical for timely treatment and recovery.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    if st.button('Predict'):
        st.write("Classifying...")
        label = predict_image(img)
        st.write(f"The image is classified as: {label}")

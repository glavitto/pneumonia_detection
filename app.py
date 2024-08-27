import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model(r'D:\luminar\cnn project\cnn_model.h5')

# Define a function to make predictions
def predict_image(img):
    img = img.resize((150, 150))  # Resize image to match model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    predictions = model.predict(img_array)
    return "NORMAL" if predictions[0][0] > 0.5 else "PNEUMONIA"

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

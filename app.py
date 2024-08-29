import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load your TFLite model
interpreter = tflite.Interpreter(model_path=r'D:\luminar\cnn project\model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to make predictions
def predict_image(img):
    img = img.resize((150, 150))  # Resize image to match model's expected input size
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run the inference
    interpreter.invoke()

    # Extract the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return "NORMAL" if output_data[0][0] > 0.5 else "PNEUMONIA"

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

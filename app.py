import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('yawn_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Define the Streamlit app
def app():
    st.title('Yawn Detection App')

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Read the image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        # Display the image
        st.image(img, caption='Uploaded Image', use_column_width=True)
        # Preprocess the image
        img = preprocess_image(img)

        # Make a prediction
        pred = model.predict(img)
        pred_label = 'Yawning' if pred[0][0] > 0.5 else 'No Yawning'
        pred_prob = pred[0][0]
        
        # Show the prediction result
        st.write(f'Prediction: {pred_label}')
        st.write(f'Probability Of Yawning: {pred_prob:.2f}')

# Run the app
if __name__ == '__main__':
    app()
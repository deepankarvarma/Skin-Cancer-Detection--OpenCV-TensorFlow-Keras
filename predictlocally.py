import cv2
import numpy as np
from keras.models import load_model

# Set the size of the input images
img_size = (224, 224)

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Open the camera stream
cap = cv2.VideoCapture(0)

# Loop through the frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    img = cv2.resize(frame, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Predict whether the frame contains a yawn or not
    prediction = model.predict(img)
    if prediction[0] > 0.5:
        label = 'Cancer'
    else:
        label = 'Non_Cancer'
    
    # Draw the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
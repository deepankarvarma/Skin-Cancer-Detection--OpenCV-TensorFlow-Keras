import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization

# Set path to the data folder
data_dir = 'Skin_Data'

# Set the size of the input images
img_size = (224, 224)

# Set the batch size for training
batch_size = 32

# Create data generators for training and validation
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_data_gen = ImageDataGenerator(rescale=1./255)

# Load the images and labels from the data folder
x_train = []
y_train = []
for label, folder_name in enumerate(['Non_Cancer/Training', 'Cancer/Training']):
    folder_path = os.path.join(data_dir, folder_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        x_train.append(img)
        y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val = []
y_val = []
for label, folder_name in enumerate(['Non_Cancer/Testing', 'Cancer/Testing']):
    folder_path = os.path.join(data_dir, folder_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        x_val.append(img)
        y_val.append(label)

x_val = np.array(x_val)
y_val = np.array(y_val)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data_gen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=val_data_gen.flow(x_val, y_val),
                    epochs=20, verbose=1)

# Save the model
model.save('skin_cancer_detection_model.h5')
# Skin Cancer Detection Model

This repository contains Python code for generating a skin cancer detection model and using it to detect skin cancer from user-inputted images or videos. The model architecture is as follows:

```python
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
```

## Dataset

The dataset used for training and evaluation can be downloaded from Kaggle: [Skin Cancer Binary Classification Dataset](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset). It provides labeled images for binary classification of skin cancer.

## Dependencies

To run the code in this repository, you'll need the following dependencies:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV

You can install the required packages using `pip`:

```shell
pip install tensorflow keras numpy opencv-python
```

## Usage

1. Clone this repository to your local machine:

```shell
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Download the Skin Cancer Binary Classification Dataset from the provided link and place it in the appropriate directory.

3. Use the provided code to train the skin cancer detection model.

4. Run the script to detect skin cancer from an image:

```shell
python predict_image.py --image path/to/your/image.jpg
```

5. Run the script to detect skin cancer from a video:

```shell
python predict_video.py --video path/to/your/video.mp4
```

Make sure to replace `path/to/your/image.jpg` and `path/to/your/video.mp4` with the actual paths to your desired image and video files, respectively.

## Results

The skin cancer detection model, trained on the Skin Cancer Binary Classification Dataset, can accurately classify skin cancer from images and videos. You can modify the code and experiment with different architectures or hyperparameters to potentially improve the performance.

## Acknowledgments

- The Skin Cancer Binary Classification Dataset used in this project was sourced from Kaggle: [Skin Cancer Binary Classification Dataset](https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset).

## License

This project is licensed under the [MIT License](LICENSE).

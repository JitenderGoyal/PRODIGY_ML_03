import os  # Import the Python os module for interacting with the operating system.
import matplotlib.pyplot as plt  # Import the matplotlib library for data visualization.
import numpy as np  # Import the NumPy library for numerical operations.
import pandas as pd  # Import the Pandas library for data manipulation.
from skimage import color, exposure, io, transform  # Import specific functions/modules from the skimage library.
from skimage.feature import hog  # Import the Histogram of Oriented Gradients (HOG) feature extraction method.
from sklearn.model_selection import train_test_split  # Import train_test_split from scikit-learn for dataset splitting.

class PreProcess:
    def __init__(self):
        # Constructor for the PreProcess class. It initializes the class, but there's no specific data passed in.
        # self.data = file_list
        pass

    def file_labelling(self, cat_path, dog_path):
        # Method for labeling cat and dog image files.
        cat_dir = os.listdir(cat_path)  # Get a list of files in the cat_path directory.
        cat_file = [os.path.join(cat_path, i) for i in cat_dir]  # Create full file paths for cat images.
        cat_df = pd.DataFrame(data=cat_file, columns=['file'])  # Create a Pandas DataFrame for cat images.
        cat_df['target'] = 0  # Assign a target label of 0 for cat images.

        dog_dir = os.listdir(dog_path)  # Get a list of files in the dog_path directory.
        dog_file = [os.path.join(dog_path, i) for i in dog_dir]  # Create full file paths for dog images.
        dog_df = pd.DataFrame(data=dog_file, columns=['file'])  # Create a Pandas DataFrame for dog images.
        dog_df['target'] = 1  # Assign a target label of 1 for dog images.

        return cat_df, dog_df  # Return DataFrames for cat and dog images.

    def test_file_labelling(self, path):
        # Method for labeling test image files.
        test_dir = os.listdir(path)  # Get a list of files in the specified path.
        test_file = [os.path.join(path, i) for i in test_dir]  # Create full file paths for test images.
        test_df = pd.DataFrame(data=test_file, columns=['Location'])  # Create a Pandas DataFrame for test images.

        return test_df  # Return the DataFrame for test images.

    def extract_hog_feature(self, image_path):
        # Method for extracting HOG features from an image.
        image = io.imread(image_path)  # Read the image from the provided path.
        # Resize the image to a consistent size (128x128 pixels).
        resized_image = transform.resize(image, (128, 128))
        # Convert the resized image to grayscale for feature extraction.
        gray_image = color.rgb2gray(resized_image)
        # Calculate HOG features with specific parameters.
        fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)
        return fd.flatten(), image_path  # Return the flattened HOG features and the image path.

    def split_dataset(self, df):
        # Method for splitting a dataset into training and testing sets.
        X = df.iloc[:, :-1]  # Get the feature columns.
        y = df.iloc[:, -1]  # Get the target column.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=100)  # Split the dataset using train_test_split.

        return X_train, X_test, y_train, y_test  # Return the training and testing sets for features and labels.

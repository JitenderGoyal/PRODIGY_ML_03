# Image Classification using Support Vector Machine (SVM) with HOG Features

## Overview
This project demonstrates image classification using Support Vector Machine (SVM) with Histogram of Oriented Gradients (HOG) features. It consists of three main components: preprocessing images using the `PreProcess` class, training the SVM model with the `Train.ipynb` notebook, and making predictions on test images using the `Test.ipynb` notebook.

## PreProcess File (pre_process.py)

### Description
The `PreProcess` class contains methods for data preprocessing, including file labeling, HOG feature extraction, and dataset splitting. Here's what each method does:

- `file_labelling(cat_path, dog_path)`: Labels image files as cat or dog, returning two DataFrames.
- `test_file_labelling(path)`: Labels test image files, returning a DataFrame.
- `extract_hog_feature(image_path)`: Extracts HOG features from an image.
- `split_dataset(df)`: Splits a dataset into training and testing sets.

## Train.ipynb

### Description
The `Train.ipynb` Jupyter notebook is used to train the SVM model and evaluate its performance. It performs the following tasks:

1. Import the `PreProcess` class and necessary libraries.
2. Label the image files as cat and dog and display the first few rows of the data.
3. Extract HOG features from a subset of cat and dog images and create DataFrames.
4. Concatenate the HOG features and create the final dataset.
5. Split the dataset into training and testing sets and scale the features.
6. Initialize an SVM classifier, train it, and save the model.
7. Calculate and display precision, recall, accuracy, F1-score, confusion matrix, and a classification report.

## Test.ipynb

### Description
The `Test.ipynb` Jupyter notebook is used to make predictions on test images and visualize the results. It performs the following tasks:

1. Import the `PreProcess` class and necessary libraries.
2. Label test image files and display the first few rows of the data.
3. Extract HOG features from test images and create a DataFrame.
4. Load the trained SVM model.
5. Make predictions on the test dataset and update the test DataFrame with the target labels.
6. Display unique target labels and visualize test images with their predictions.



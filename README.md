# ASL-Recognition-Deep-Learning
This repository is for a project that uses deep learning techniques to recognize American Sign Language (ASL) gestures. It aims to facilitate communication for individuals by accurately recognizing ASL signs using computer vision.

# Datasets
The datasets folder contains three zip files (A.zip, B.zip, C.zip) with images of ASL gestures. The images are organized into three folders (A, B, C) based on the type of gesture. The folder structure is as follows:

```
datasets
├── A
│   ├── 0
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   ├── 1
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   └── ...
├── B
│   ├── 0
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   ├── 1
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   └── ...
├── C
│   ├── 0
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   ├── 1
│   │   ├── img_0.png
│   │   ├── img_1.png
│   │   └── ...
│   └── ...
└── sign_language.py
```

The sign_language.py file contains a function load_data that loads the datasets and returns them as NumPy arrays. The function takes the following arguments:

- container_path (optional): the path to the folder containing the datasets. The default value is 'datasets'.
- folders (optional): a list of folder names (A, B, C) to load the datasets from. The default value is ['A', 'B', 'C'].
- size (optional): the maximum number of images to load from each folder. The default value is 2000.
- test_split (optional): the fraction of the datasets to use for testing. The default value is 0.2.
- seed (optional): the seed for the random number generator. The default value is 0.

To use the load_data function, you can call it with the desired arguments and assign the return value to a variable, like this:

`(x_train, y_train), (x_test, y_test) = load_data()`

This will load the datasets and split them into training and testing sets. The training set will be stored in the x_train and y_train variables, and testing set will be stored in the x_test and y_test variables. The x_train and x_test variables will contain the images as NumPy arrays, and the y_train and y_test variables will contain the corresponding labels as NumPy arrays.

To use the datasets in your code, you can access the images and labels using the x_train, y_train, x_test, and y_test variables. For example, you can use the x_train and y_train variables to train a machine learning model, and you can use the x_test and y_test variables to evaluate the model's performance. Here is an example of how you could use the load_data function to load the datasets and train a machine learning model:

```
### Load the datasets
(x_train, y_train), (x_test, y_test) = load_data()

### Train a machine learning model on the training set
model = ...
model.fit(x_train, y_train)

### Evaluate the model's performance on the testing set
loss, accuracy = model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)
```

This code will load the datasets, train a machine learning model on the training set, and evaluate the model's performance on the testing set. The test accuracy will be printed to the console.

# Dependencies
To run the code in this repository, you will need the following dependencies:

- Python 3.6 or later
- TensorFlow 2.4 or later
- NumPy
- Matplotlib

You can install the dependencies using pip:

```pip install tensorflow numpy matplotlib```

# Getting Started
To get started with the project, follow these steps:

- Clone the repository to your local machine.
- Install the dependencies using pip.
- Open the Google Colab notebook and follow the instructions to load the datasets and train a machine learning model.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
This project was inspired from (Alexis Cook)[https://www.kaggle.com/alexisbcook].

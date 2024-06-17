import random
import numpy as np
from keras.utils import np_utils, to_categorical
from keras.preprocessing import image
from os import listdir
from os.path import isdir, join


def load_data(container_path='datasets', folders=['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0):
    """
    Loads sign language dataset.

    Args:
        container_path (str): Path to the folder containing the dataset.
        folders (list): List of folders containing the images.
        size (int): Number of images to load.
        test_split (float): Fraction of data to be used for testing.
        seed (int): Random seed.

    Returns:
        tuple: A tuple containing the training and testing data.
    """
    
    # List to store filenames and labels
    filenames, labels = [], []

    # Iterate over the folders
    for label, folder in enumerate(folders):
        # Get the path to each folder
        folder_path = join(container_path, folder)
        # Get the list of images in each folder
        images = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        # Extend the filenames and labels lists
        labels.extend(len(images) * [label])
        filenames.extend(images)
    
    # Shuffle the data
    random.seed(seed)
    data = list(zip(filenames, labels))
    random.shuffle(data)
    # Select the specified number of images
    data = data[:size]
    # Unzip the filenames and labels
    filenames, labels = zip(*data)

    # Convert the filenames to tensors
    x = paths_to_tensor(filenames).astype('float32')/255
    # Store the one-hot targets
    y = np.array(labels)

    # Split the data into training and testing sets
    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    # Return the training and testing data
    return (x_train, y_train), (x_test, y_test)


def path_to_tensor(img_path, size):
    """
    Loads an image from the given path, resizes it to the specified size,
    converts it to a 3D tensor, and adds a batch dimension to it.

    Args:
        img_path (str): The path to the image file.
        size (int): The target size of the image.

    Returns:
        numpy.ndarray: The 4D tensor representing the image with a batch dimension.
    """
    # Loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(size, size))

    # Converts PIL.Image.Image type to 3D tensor
    x = image.img_to_array(img)

    # Adds a batch dimension to the 3D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, size=50):
    list_of_tensors = [path_to_tensor(img_path, size) for img_path in img_paths]
    return np.vstack(list_of_tensors)

"""
    num_types = len(data['target_names'])
    targets = np_utils.to_categorical(np.array(data['target']), num_types)
"""

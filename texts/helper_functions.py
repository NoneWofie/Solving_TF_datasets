import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import zipfile
import pathlib
import matplotlib.image as mpimg
import random
import datetime

####################
# Getting Datasets #
####################

#
# Extracting zip files to dataset folder
#


def extract_zip_file_to_folder(zip_names_list, folder_name="dataset"):
    """
    Extracts zip files on the root directory, and places it in a folder.

    Parameters
    ----------
    folder_name (str): folder name to store the unzipped files in.
    zip_names_list (list): A list of strings that contains zipped file names.
    """

    for each_zip_file in zip_names_list:
        zip_ref = zipfile.ZipFile(each_zip_file, "r")
        zip_ref.extractall(f"./{folder_name}/")

######################
# Exploring Datasets #
######################

#
# Exploring number of directories and files in a folder
#


def exploring_num_directories_files(folder_name="dataset"):
    """
    Explores the number of directories & files in a folder.

    Parameters
    ----------
    folder_name (str): folder name to explore from.
    """
    for dirpath, dirnames, filenames in os.walk(folder_name):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

#
# Number of Image classes based from the directories
#


def num_of_image_classes_from_directory(folder_name="./dataset/train/"):
    """
    Gathers the number of different classes based on the folder structure.

    Parameters
    ----------
    folder_name (str): A folder which contain all the different categories e.g. "dataset/train/"
    """
    data_dir = pathlib.Path(
        folder_name)  # turn our training path into a Python path
    # created a list of class_names from the subdirectories
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

    print(class_names)

    return class_names

#
# Viewing random Images from dataset
#


def view_random_image(target_dir="./datasets/train/", target_class="class_name"):
    """
    Views random images on the image dataset.

    Parameters
    ----------
    target_dir (str): A target folder where different categorical exists
    target_class (str): A target class to view an image
    """
    # Setup target directory (we'll view images from here)
    target_folder = target_dir+target_class

    # Get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    # plt.xlabel(f"Image shape: {img.shape}")

    plt.xticks([])
    plt.yticks([])

    # plt.axis("off")

    # print(f"Image shape: {img.shape}") # show the shape of the image

    return img

#############
# Callbacks #
#############

def create_tensorboard_callback(dir_name, experiment_name):
    """
    To create a tensorboard callback for model training

    Parameters
    ----------
    dir_name (str): A folder where all experiments are stored
    experiment_name (str): An experiment name
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
      )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

##############
# Evaluation #
##############

#
# Plotting loss curves
#

def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    
    Parameters
    ----------
    history (dict): The return history object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

#
# Loading and preparing an image for prediction
#
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).

  Parameters
  ----------
  filename (str): Filename of a file in the root directory
  img_shape (int): size of the image that the model takes in
  """
  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels 
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.

  return img
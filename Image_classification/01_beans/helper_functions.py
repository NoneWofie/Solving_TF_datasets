import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 

import os
import zipfile
import pathlib
import matplotlib.image as mpimg
import random

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
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

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
    data_dir = pathlib.Path(folder_name) # turn our training path into a Python path
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories

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

#
# View random images per class
#
def view_random_image_per_class(target_dir="./datasets/train/", classes=[], ncols=6, figsize=(14, 7)):
    """
    View random images on all classes per row.

    Parameters
    ----------
    target_dir (str): A target folder where different categorical exists
    classes (list): A list of class names
    ncols (int): Number of random images to be viewed per row
    figsize (tuple) : How big you want the plot to be
    """
    fig, axs = plt.subplots(len(classes), ncols, figsize=figsize)

    for index_row, each_row_ax in enumerate(axs):
        #Choose a class folder
        target_folder = target_dir + classes[index_row]

        # Get a random image path
        random_image = random.sample(os.listdir(target_folder), ncols)

        for index_col, each_column_ax in enumerate(each_row_ax):

            img = mpimg.imread(target_folder + "/" + random_image[index_col])

            each_column_ax.imshow(img)
            each_column_ax.set_xticks([])
            each_column_ax.set_yticks([])

            if index_col == 0:
                each_column_ax.set_ylabel(classes[index_row], fontdict={'fontsize': 16, 'fontweight': 'medium'})
            
    plt.tight_layout()
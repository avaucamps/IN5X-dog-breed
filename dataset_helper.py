from __future__ import print_function
import pandas as pd
import shutil
import os
import sys
import zipfile
import numpy as np
from constants import train_data_dir, validation_data_dir, default_train_dir, labels_path, labels_zip_path

def prepare_dataset():
    ''' 
    Extracts dataset if needed. 
    Creates and populates training and validation directories if needed.
    '''
    unzip_dataset()
    labels = pd.read_csv(labels_path)
    cross_validation_indexes = get_cross_validation_indexes(len(labels))
    
    if not os.path.exists(train_data_dir):
        map_training_pictures_to_labels(labels, cross_validation_indexes)
    if not os.path.exists(validation_data_dir):
        map_validation_pictures_to_labels(labels, cross_validation_indexes)

def unzip_dataset():
    if not os.path.exists(labels_path) and os.path.exists(labels_zip_path):
        labels_zip = zipfile.ZipFile(labels_zip_path)
        labels_zip.extractall()
    if not os.path.isdir('train'):
        train_images_zip = zipfile.ZipFile(default_train_dir + '.zip')
        train_images_zip.extractall()
    
def map_training_pictures_to_labels(labels, cross_validation_indexes):
    create_directory(train_data_dir)

    index = 0
    for filename, class_name in labels.values:
        if index in cross_validation_indexes:
            index = index + 1
            continue

        # Create subdirectory with `class_name`
        create_directory(train_data_dir + class_name)

        src_path = default_train_dir + filename + '.jpg'
        dst_path = train_data_dir + class_name + '/' + filename + '.jpg'
            
        index = index + 1

        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                .format(src_path, dst_path, sys.exc_info())) 

def map_validation_pictures_to_labels(labels, cross_validation_indexes):
    create_directory(validation_data_dir)   

    index = 0
    for filename, class_name in labels.values:
        if index not in cross_validation_indexes:
            index = index + 1
            continue

        # Create subdirectory with `class_name`
        create_directory(validation_data_dir + class_name)

        src_path = default_train_dir + filename + '.jpg'
        dst_path = validation_data_dir + class_name + '/' + filename + '.jpg'
            
        index = index + 1

        try:
            shutil.copy(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                .format(src_path, dst_path, sys.exc_info())) 

def create_directory(name):
    if not os.path.exists(name):
       os.mkdir(name)

def get_cross_validation_indexes(dataset_size, validation_set_percentage = 0.2):
    number_of_values = int(dataset_size * validation_set_percentage)
    indexes = np.random.permutation(dataset_size)

    return indexes[0:number_of_values]
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys
import zipfile
import numpy as np

default_train_dir = 'train/'
custom_train_dir = 'train_custom/'
custom_validation_dir = 'validation_custom/'

def prepare_dataset():
    ''' Method to be called at the beginning to prepare the dataset. 
    If the dataset is still in zip files it is extracted.
    If the training and validation directories have not been created, they are created and the training
    and validation images are copied in those directories.
    '''
    unzip_dataset()
    labels = pd.read_csv('labels.csv')
    cross_validation_indexes = get_cross_validation_indexes(len(labels))
    
    if not os.path.exists('train_custom'):
        map_training_pictures_to_labels(labels, cross_validation_indexes)
    if not os.path.exists('validation_custom'):
        map_validation_pictures_to_labels(labels, cross_validation_indexes)

def unzip_dataset():
    if not os.path.exists('labels.csv'):
        labels_zip = zipfile.ZipFile('labels.csv.zip')
        labels_zip.extractall()
    if not os.path.exists('train'):
        train_images_zip = zipfile.ZipFile('train.zip')
        train_images_zip.extractall()
    
def map_training_pictures_to_labels(labels, cross_validation_indexes):
    create_directory(custom_train_dir)

    index = 0
    for filename, class_name in labels.values:
        if index in cross_validation_indexes:
            index = index + 1
            continue

        # Create subdirectory with `class_name`
        create_directory(custom_train_dir + class_name)

        src_path = default_train_dir + filename + '.jpg'
        dst_path = custom_train_dir + class_name + '/' + filename + '.jpg'
            
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
    create_directory(custom_validation_dir)   

    index = 0
    for filename, class_name in labels.values:
        if index not in cross_validation_indexes:
            index = index + 1
            continue

        # Create subdirectory with `class_name`
        create_directory(custom_validation_dir + class_name)

        src_path = default_train_dir + filename + '.jpg'
        dst_path = custom_validation_dir + class_name + '/' + filename + '.jpg'
            
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
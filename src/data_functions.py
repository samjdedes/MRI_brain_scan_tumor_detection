import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def get_img_paths(base_dir):
    """
    Retrieves paths to all files given a base directory.
    
    Parameters:
        base_dir (string): relative path to the directory with folders containing images

    Returns:
        paths (list of tuples): relative paths (string) to files with identifiers (int)
    """
    # instantiate empty lists
    absent_paths = []
    present_paths = []

    # loop through each dir and file in root (base_dir) given
    for root, dirs, files in os.walk(base_dir):
                   
        # if files present, save path and append to list with classification/identifier
        if files:
            for file in files:
                full_path = os.path.join(root, file)
                
                if 'NO' in full_path:
                    absent_paths.append((full_path, 0))

                elif 'YES' in full_path:
                    present_paths.append((full_path, 1))

    return absent_paths, present_paths


def get_data(input_data, img_size=150):
    """
    Retrieves image data with classification/type, and path

    Parameters:
        input_data (list of tuples): paths leading to image data to be loaded and classification/type

    Returns:
        data (np.array of tuples): image data (np.array) with classification/type (int), and path (string)
    """

    data = []

    for i in range(len(input_data)):
        path = input_data[i][0]
        class_num = input_data[i][1]

        #         for img in os.listdir(path):

        try:
            img_arr = cv2.imread(path)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
            data.append([resized_arr, class_num, path])

        except Exception as e:
            print(f'{e} on path {path}')

    return np.array(data)

def remove_duplicates(data):
    '''
    Removes duplicate entries from list of data.
    
    Parameters:
        data (list of tuples): contains image data (np.array), classification (int) to compare to others within input data
        
    Returns:
        unique_list (np.array): array of unique image data (np.array) with classifcations (int)
        duplicate_list (np.array): array of duplicate image data (np.array) removed from input data
    
    '''

    unique_list = []
    duplicate_list = []
    # loop through original images
    for image in data:
        img = image[0]
        
        # informs function to append image
        is_unique = True
        
        # loop through new list
        for unique_image in unique_list:
            unique_img = unique_image[0]
            
            # check existing entries to new entry
            if (img == unique_img).all():
                is_unique = False
                print(f'duplicate found:{img[0][25:26]} and {unique_img[0][25:26]}')
                break
                
        # add to unique list if unique
        if is_unique:
            unique_list.append(image)
        
        else:
            duplicate_list.append(image)
            
    return np.array(unique_list), np.array(duplicate_list)
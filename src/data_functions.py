import os
import math 
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.segmentation import mark_boundaries
import lime.lime_image as li


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
                break
                
        # add to unique list if unique
        if is_unique:
            unique_list.append(image)
        
        else:
            duplicate_list.append(image)
            
    return np.array(unique_list), np.array(duplicate_list)


def plot_metrics(model_history, filename='default', path=None):
    '''
    Plots model performance over epochs and saves as .png
    
    Parameters:
        model_history (dict): history of a model
        *kwargs
        is_dict (bool): specify if file is dict type
        filename (string): appends file with name 'default' unless specified or path kwarg specified. Set to None to prevent file from saving
        path (string): full path to save file. File type may be specified
        
    '''
    # get history from callback if dict not specified
    try:
        model_history = dict(model_history)
        
    except:
        model_history = model_history.history
        
    # get epochs from history from model
    n_epochs = len(model_history['loss'])
    epochs = [i for i in range(n_epochs)]
    
    # plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('fivethirtyeight')
    plt.rcParams['text.color'] = 'k'
    plt.rcParams['axes.facecolor'] = 'F0F0F0'

    fig , ax = plt.subplots(1,2)
    train_acc = model_history['accuracy']
    train_loss = model_history['loss']
    val_acc = model_history['val_accuracy']
    val_loss = model_history['val_loss']
      
    
    fig.set_size_inches(20,10)

    ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
    ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['left'].set_color('black')
    ax[0].tick_params(axis='x', colors='black')
    ax[0].tick_params(axis='y', colors='black')
    ax[0].yaxis.label.set_color('black')
    ax[0].xaxis.label.set_color('black')

    ax[1].plot(epochs , train_loss , 'g-o' , label = 'Testing Loss')
    ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Testing & Validation Loss")
    ax[1].spines['bottom'].set_color('black')
    ax[1].spines['left'].set_color('black')
    ax[1].tick_params(axis='x', colors='black')
    ax[1].tick_params(axis='y', colors='black')
    ax[1].yaxis.label.set_color('black')
    ax[1].xaxis.label.set_color('black')
    
    if path:
        plt.savefig(f'{path}', transparent=True)
    
    elif filename:
        plt.savefig(f'figures/plot_metrics_{filename}', transparent=True)
        
    plt.show()


def plot_confusion_matrix(model, X, Y, labels=['No Tumor', 'Tumor'], filename='default', path=None):
    '''
    Creates classification report, plots, and saves confusion matrix using the existing model (model), 
        test data (X_test), and test classifications (Y_test).
        
    Parameters: 
        model (keras model): trained model used to make predictions of data
        X (np.array): image data used to make model predictions
        Y (np.array): image classifications/ target (int) used to determine accuracy of predictions
        *kwargs
        labels (list): xlabels and ylabels for plot of confusion matrix, and target_names used in classification report
        filename (string): appends file with name 'default' unless specified or path kwarg specified. Set to None to prevent file from saving
        path (string): full path to save file. File type may be specified
    '''
    
    # get predictions
    predictions = (model.predict(X) > 0.5).astype('int32')
    print(classification_report(Y, predictions, target_names=labels))
    
    #create confusion matrix
    cm = confusion_matrix(Y, predictions)
    print(cm)
    
    cm = pd.DataFrame(cm, index=['0', '1'], columns=['0', '1'])
    
    # plot confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, cmap='Greys', linecolor='grey', linewidth=1, annot=True, fmt='', xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Class')    
    plt.ylabel('Image Class')
    
    if path:
        plt.savefig(f'{path}', transparent=True)
    
    elif filename:
        plt.savefig(f'figures/plot_confusion_matrix_{filename}', transparent=True)
        
    plt.show()
    
    
# def lime_image(model, image, min_superpixels=1, max_superpixels=10, positive_only=False, negative_only=False, hide_rest=False,\
#                 filename='default', path=None, figsize=(15,15), axis='off', suptitle='Region Analysis with LIME'):
#     '''
#     Create a visual of the inner workings of the image processing neural network using LimeImageExplainer from lime.lime_image.
#         It does this by separating the image into various regions known as "superpixels" and judging model's performance
#         with and without these superpixels on the image.
    
#     Parameters:
#         model (keras sequantial model): model used in analysis of image
#         image (np.array): image to be analyzed
#         *kwargs
#         min_superpixels (int): minimum number of regions in LIME analysis of model's classification. Default 1
#         max_superpixels (int): maxmum number of regions in LIME analysis of model's classification. Default 10
#         positive_only (bool): indicate whether to include superpixels correlated to correct classification. Default False
#         negative_only (bool): indicate whether to include superpixels correlated to incorrect classification. Default False
#         hide_rest (bool): if set to True, hides parts of image not included in superpixels
#         filename (string): appends file with name 'default' unless specified or path kwarg specified. Set to None to prevent file from saving
#         path (string): full path to save file, file type may be specified. Default None
#         figsize (tuple): size of figure to be saved. Default (15, 15)
#         axis (string): turn axis off. Default 'off'
#         suptitle (string): main title of plot. Default 'Different Features Analyzed by Model'
#     '''
    
#     # instantiate image explainer
#     explainer = li.LimeImageExplainer()
    
#     # difference
#     diff = max_superpixels - min_superpixels
    
#     # calculate shape of figure
#     columns = int(np.ceil((diff)**0.5))
#     rows = int(np.ceil(diff/columns))
    
#     # get grid params
#     grid_max = rows*columns
#     grid_diff = grid_max - diff
    
#     # instantiate plot to populate with explained images
#     fig, ax = plt.subplots(rows, columns, figsize=figsize)
#     ax = ax.flatten()
#     m_end = diff // rows
#     n_end = diff % (columns)
    
#     for i in range(diff):
#         k = i + min_superpixels
        
#         # analyze image and create mask
#         explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
#         temp, mask = explanation.get_image_and_mask(0, num_features=k, positive_only=positive_only, negative_only=negative_only, hide_rest=hide_rest)

#         # plot results
#         ax[i].imshow(mark_boundaries(temp/2 + 0.5, mask))
#         ax[i].axis(axis)
#         ax[i].set_title(f'# of Superpixels: {k}')            

    
#     if grid_diff:
#         for j in range(diff, grid_max):
#             ax[j].axis(axis)
            
    
#     fig.suptitle(suptitle)
    
#     if path:
#         plt.savefig(f'{path}', transparent=True)
    
#     elif filename:
#         plt.savefig(f'figures/plot_confusion_matrix_{filename}', transparent=True)
        
#     plt.show()


def lime_image(model, image, min_superpixels=1, max_superpixels=10, positive_only=False, negative_only=False, hide_rest=False,\
                filename='default', path=None, figsize=(15,10), axis='off', suptitle='Different Features Analyzed by Model'):
    '''
    Create a visual of the inner workings of the image processing neural network using LimeImageExplainer from lime.lime_image.
        It does this by separating the image into various regions known as "superpixels" and judging model's performance
        with and without these superpixels on the image.
    
    Parameters:
        model (keras sequantial model): model used in analysis of image
        image (np.array): image to be analyzed
        *kwargs
        min_superpixels (int): minimum number of regions in LIME analysis of model's classification. Default 1
        max_superpixels (int): maxmum number of regions in LIME analysis of model's classification. Default 10
        positive_only (bool): indicate whether to include superpixels correlated to correct classification. Default False
        negative_only (bool): indicate whether to include superpixels correlated to incorrect classification. Default False
        hide_rest (bool): if set to True, hides parts of image not included in superpixels
        filename (string): appends file with name 'default' unless specified or path kwarg specified. Set to None to prevent file from saving
        path (string): full path to save file, file type may be specified. Default None
        figsize (tuple): size of figure to be saved. Default (15, 15)
        axis (string): turn axis off. Default 'off'
        suptitle (string): main title of plot. Default 'Different Features Analyzed by Model'
    '''
    
    # instantiate image explainer
    explainer = li.LimeImageExplainer()
    
    # difference
    diff = max_superpixels - min_superpixels
    
    # calculate shape of figure
    columns = int(np.ceil((diff)**0.5))
    rows = int(np.ceil(diff/columns))
    
    # get grid params
    grid_max = rows*columns
    grid_diff = grid_max - diff
    
    print(f'rows: {rows}  columns: {columns}')
    
    # instantiate plot to populate with explained images
    fig, ax = plt.subplots(rows, columns, figsize=figsize)
    print(f'len(ax) = {len(ax)}')
    ax = ax.flatten()
    m_end = diff // rows
    n_end = diff % (columns)
    print(f'm_end: {m_end}  n_end: {n_end}')
    
    for i in range(diff):
        k = i + min_superpixels
        
        # analyze image and create mask
        explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(0, num_features=k, positive_only=positive_only, negative_only=negative_only, hide_rest=hide_rest)

        # plot results
        ax[i].imshow(mark_boundaries(temp/2 + 0.5, mask))
        ax[i].axis(axis)
        ax[i].set_title(f'# of Superpixels: {k}')            

    
    if grid_diff:
        for j in range(diff, grid_max):
            ax[j].axis(axis_off)
            
    
    fig.suptitle(suptitle)
    
    if path:
        plt.savefig(f'{path}', transparent=True)
    
    elif filename:
        plt.savefig(f'figures/plot_confusion_matrix_{filename}', transparent=True)
        
    plt.show()
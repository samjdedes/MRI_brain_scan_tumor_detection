# Tumor Detection: Analyzing MRI Brain Scans

## TABLE OF CONTENTS

### Notebooks

[Exploratory Data Analysis](notebooks/exploratory)

[Final Report Notebook](notebooks/report)


### Reports
[Executive Summary](reports/presentation)

[Figures](reports/figures)


### Data

[How to access data](data)


### SRC

[Custom Functions](src)

### ReadMe

[Read Me](README.md)


# Project Goal and Background
Project development completed with Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology. The goal is to develop and train a Convolutional Neural Network (CNN) machine learning model to identify the presence of tumors in MRI brain scans.


## Business Understanding 

According to the [National Institute of Health](https://www.ninds.nih.gov/Disorders/Patient-Caregiver-Education/Hope-Through-Research/Brain-and-Spinal-Tumors-Hope-Through#definition), "[brain tumors] occur when something goes wrong with genes that regulate cell growth, allowing cells to grow and divide out of control... Depending on its type, a growing tumor may not cause any symptoms or can kill or displace healthy cells or disrupt their function" (Brain and Spinal Cord Tumors, 2020).

Brain tumors are diagnosed fist by a neurological exam, and then through imaging methods including CT and MRI scans. If necessary, a biopsy is done to confirm a diagnosis (Brain and Spinal Cord Tumors, 2020). A biopsy is a surgical procedure where a small sample of tissue is extracted. Depending on the location of the suspected tumor, this can be dangerous to the patient, or impossible to perform if in a particularly sensitive area.

The stakeholders here are the doctors and MRI technicians treating pediatric with suspected brain tumors.


## Data Understanding

The data used includes thousands of images of MRI brain scans. The data was sourced from two Kaggle datasets: [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) and [Kaggle](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).

In this project, a scan with a tumor was considered Class 0, and a scan with a tumor present was considered Class 1. As a result, recall score was the prioritized along with accuracy in this type of neural network, as the effects of false negative can be much more harmful than a false positive.


## Data Preparation
After retrieving the data from the dataset it is important to classify images according to their labels. Because we are running a CNN, a supervised learning machine method, labels need to be attributed to images to help in the process of reducing loss, and increasing recall and accuracy. After this is done, each image is converted into an 1-D array, and divided by 255. The values of the grayscale pixels range from 0-255. Dividing each pixel by 255 normalizes our grayscale to values between 0-1, and also helps our CNN algorithm converge faster.

![class imbalance](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/image_classification.png)
Next we needed to address our class imbalance. The first was to use a keras preprocessing function called ImageDataGenerator. ImageDataGenerator takes a batch of images used for training, applies a series of random transformation to each image in the batch (including random rotation, resizing and shearing), and replaces the original batch with the new randomly transformed batch. This effectively allows us to expand the training dataset in order to improve the performance and ability of the model to generalize.


## Modeling

Artificial intelligence (AI) Deep Neural Network makes it possible to process images in the form of pixels as input and to predict the desired classification as output. The development of Convolutional Neural Network (CNN) layers trains a model for significant gains in the ability to classify images and detect objects in a picture.  Multiple processing layers use image analysis filters, or convolutions as the model is trained.

The convolutional layers in the network along with filters help in extracting the spatial and temporal features in an image. The layers have a weight-sharing technique, which helps in reducing computation efforts.

A Convolution Neural Network (CNN) is built on three broad strategies:

  1)Learn features using Convolution layer

  2)Reduce computational costs by down sample the image and reduce dimensionality using Max-Pooling(subsampling)

  3)Fully connected layer to equip the network with classification capabilities


![CNN Visual](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/cnn.jpg)


![recall/accuracy](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/Model1_Epoch12_Batch32%20%20model_accuracy_recall.png)


![lime visuals](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/Sample_True_Predictions_5.png)


## Evaluation


## Potential Next Steps

Our next steps would be to classify images by location of tumor, and malignant or benign. Furthermore, there is a potential for early detection of tumors if labelled scans of undiagnosed brain tumors can be incorporated into the model.


# Appendix
## Repository Navigation

Below visualizes the structure of this repository.

MRI_brain_scan_tumor_detection
(Project Folder)
    |
    •README.md (Current file. Markdown file Containing Information on Project Purpose, Process, and Findings)
    |
    |       
    ├ data (Folder Containing Reference Data)
    |    |
    |    └ README_MRI_Data.txt (Text file containing sources of image data)
    |
    |
    ├ notebooks (Folder Containing Notebooks Used as Basis of Analysis)
    |    |
    |    ├ exploratory
    |    |   |
    |    |   ├ EDA.ipynb (Jupyter Notebook containing data exploration process, initial model development, and creation of visualizations)
    |    |   |
    |    |   └ make_pictures_EDA.ipynb (Jupyter Notebook containing process of creating .jpg images from .nii.gz medical imaging files, functions not yet completed)
    |    |
    |    └ report
    |        |
    |        └ index.ipynb (Jupyter Notebook calling functions from .py that import and clean data, create model and visualizations)
    |
    |
    ├ report (Folder Containing Finalized Notebooks, Presentation PDF, and Visualizations)
    |     |
    |     ├ final_notebook.pdf (PDF of final code, calls functions from .py that import and clean data, create model and visualizations)
    |     |
    |     └ figures (Folder containing all images created by index.ipynb used in final presentation)
    |
    |
    └ src (Folder containing custom functions used to import and clean data, create model and visualizations)
          |
          ├ __init__.py (Python file that allows python files in this directory to be treated as python modules)
          |
          └ data_functions.py (Python file containing all custom data functions called in index.ipynb)


## References

Brain and Spinal Cord Tumors. (2020, November 13). National Institute of Health. https://www.ninds.nih.gov/Disorders/Patient-Caregiver-Education/Hope-Through-Research/Brain-and-Spinal-Tumors-Hope-Through#definition

Brain Tumors. (2020, September 18). US National Library of Medicine. https://medlineplus.gov/braintumors.html

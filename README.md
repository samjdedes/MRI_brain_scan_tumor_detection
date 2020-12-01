# Tumor Detection: Analyzing MRI Brain Scans

This is the project repository for a machine learning model used to identify the presence of tumors in MRI brain scans with Convolutional Neural Networks.

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

Brain tumors are diagnosed first by a neurological exam, and then through imaging methods including MRI scans. Currently, images are analyzed my MRI technicians before being sent to the doctor for a final analysis. If necessary, a biopsy is done to confirm a diagnosis. A biopsy is a surgical procedure where a small sample of tissue is extracted. Depending on the location of the suspected tumor, this can be dangerous to the patient, or impossible to perform if in a particularly sensitive area.

With advances in image classification techniques, preliminary analyses can be aided by computers through algorithms like those created in this project. Furthermore, this can reduce the need for potentially dangerous biopsies for diagnosis can be reduced, allowing doctors and patients to focus on the next step, treatment. Doctors, patients, and MRI technicians stand to benefit from classification algorithms.


## Data Understanding

Thousands of MRI brain scans were used in this project. The scans were sourced from two Kaggle datasets: [Kaggle 2018](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) and [Kaggle 2020](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri).
Included in these scans are brains with and without brain tumors present, of various section and scan types. The three types of sections included are frontal, medial, and horizontal. ![Section Types](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/mri_axis2_pd500px.jpg) (Technische Universität München, n.d.)

Various scan types include Proton Density and Transverse Magnetization. Different types of scans are useful for detecting different types of tissue in different regions of the brain. Some tissue types are more visible under contrast material, while others propagate magnetization differently.
           Proton Density Scan            |  Transverse Magnetization (Type 2) Scan
:----------------------------------------:|:----------------------------------------:
![Scan Type PD](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/PD.gif) | ![Scan Type T2](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/T2.gif)

In this project, a scan with a tumor was considered Class 0, and a scan with a tumor present was considered Class 1. As a result, recall score was the prioritized along with accuracy in this type of neural network, as the effects of false negative can be much more harmful than a false positive.


## Data Preparation

After sourcing data, it was loaded, duplicates removed, and resized to work with the format of Convolutional Neural Network.
To create a more robust model, ImageDataGenerator was used to create altered images used in tandem with original images.


## Modeling

Artificial intelligence (AI) Deep Neural Network makes it possible to process images in the form of pixels as input and to predict the desired classification as output. The development of Convolutional Neural Network (CNN) layers trains a model for significant gains in the ability to classify images and detect objects in a picture.  Multiple processing layers use image analysis filters, or convolutions as the model is trained.

The convolutional layers in the network along with filters help in extracting the spatial and temporal features in an image. The layers have a weight-sharing technique, which helps in reducing computation efforts.

A Convolution Neural Network (CNN) is built on three broad strategies:

  1) Learn features using Convolution layer

  2) Reduce computational costs by down sample the image and reduce dimensionality using Max-Pooling(subsampling)

  3) Fully connected layer to equip the network with classification capabilities

The performance of the model is evaluated using Recall and Accuracy metrics calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Recall calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive), in other words recall means the percentage of a pneumonia correctly identified. More accurate model lead to make better decision. The cost of errors can be huge but optimizing model accuracy mitigates that cost.

![Confusion Matrix](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/plot_confusion_matrix_final.png)

## Potential Next Steps

The next steps will be to classify images by location of tumor, and malignant or benign. Furthermore, there is a potential for early detection of tumors if labelled scans of undiagnosed brain tumors can be incorporated into the model.


# Appendix
## Repository Navigation

Below visualizes the structure of this repository.

```
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
```

## References

Brain and Spinal Cord Tumors. (2020, November 13). National Institute of Health. https://www.ninds.nih.gov/Disorders/Patient-Caregiver-Education/Hope-Through-Research/Brain-and-Spinal-Tumors-Hope-Through#definition

Brain Tumors. (2020, September 18). US National Library of Medicine. https://medlineplus.gov/braintumors.html

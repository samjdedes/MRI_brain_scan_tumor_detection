# Tumor Detection: Analyzing MRI Brain Scans

This is the project repository for a machine learning model used to identify the presence of tumors in MRI brain scans with Convolutional Neural Networks.

## TABLE OF CONTENTS

### Notebooks

 - [Exploratory Data Analysis](notebooks/exploratory/EDA.ipynb)

 - [Final Report Notebook](notebooks/report/index.ipynb)


### Reports
 - [Executive Summary](report/presentation_slides.pdf)

 - [Figures](report/figures)


### Data

 - [Data Access Readme](data/README_MRI_Data.txt)


### SRC

 - [Custom Functions](src/data_functions.py)

### ReadMe

 - [Read Me](README.md)


# Project Goal and Background

The goal is to develop and train a Convolutional Neural Network (CNN) machine learning model to identify the presence of tumors in MRI brain scans. Project development completed with Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology.

The model classifies MRI scans of patients with and without tumors, like those below. <br>

![Scan Comparison](report/figures/scan_comparison.jpg)


## Business Understanding

According to the [National Institute of Health](https://www.ninds.nih.gov/Disorders/Patient-Caregiver-Education/Hope-Through-Research/Brain-and-Spinal-Tumors-Hope-Through#definition), "[brain tumors] occur when something goes wrong with genes that regulate cell growth, allowing cells to grow and divide out of control... Depending on its type, a growing tumor may not cause any symptoms or can kill or displace healthy cells or disrupt their function" (Brain and Spinal Cord Tumors, 2020).

Brain tumors are diagnosed first by a neurological exam, and then through imaging methods including MRI scans. Currently, images are analyzed by MRI technicians before being sent to the doctor for a final analysis. If necessary, a biopsy is done to confirm a diagnosis. A biopsy is a surgical procedure where a small sample of tissue is extracted. Depending on the location of the suspected tumor, this can be dangerous to the patient, or impossible to perform if it's in a particularly sensitive area.

With advances in image classification techniques, preliminary analyses can be aided by computers through algorithms like those created in this project. This can reduce the need for potentially dangerous biopsies, allowing doctors and patients to focus on the next step, treatment. Doctors, patients, and MRI technicians stand to benefit from classification algorithms.


## Data Understanding

Thousands of MRI brain scans were used in this project. The scans were sourced from two Kaggle datasets and BrainDevelopment.org: [Kaggle 2018](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection), [Kaggle 2020](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri), and [BrainDevelopment.org](https://brain-development.org/ixi-dataset/).
Included in these scans are brains with and without brain tumors present, of various section and scan types. The three types of sections included are frontal, medial, and horizontal. <br> ![Section Types](report/figures/mri_axis2_pd500px.jpg)
(Technische Universität München, n.d.)

Various scan types include Proton Density and Transverse Magnetization. Different types of scans are useful for detecting different types of tissue in different regions of the brain. Some types of brain tissue are more visible under contrast material and others propagate magnetization differently.
           Proton Density Scan            |  Transverse Magnetization (Type 2) Scan
:----------------------------------------:|:----------------------------------------:
![Scan Type PD](report/figures/PD.gif)    | ![Scan Type T2](report/figures/T2.gif)

In this project, a scan without a tumor was considered Class 0, and a scan with a tumor present was considered Class 1. As a result, recall score was the prioritized along with accuracy in this type of neural network, as the effects of false negative can be much more harmful than a false positive.


## Data Preparation

After sourcing data, paths to the images were saved and used to split data into train, validation, and testing images. Using the paths, images were loaded using openCV and duplicates were removed with custom functions. The images were then resized and reshaped to work with the format of the employed Convolutional Neural Network.
To create a more robust model ImageDataGenerator was explored to create slightly altered images used in tandem with original images while training the model, however, after testing several models it proved to reduce the accuracy and recall of the model and was not used in the final iteration.
.


## Modeling

A Convolutional Neural Network makes it possible to process images as the input and predict the a classification as the output. The development of Convolutional Neural Network (CNN) layers trains a model for significant gains in the ability to classify images and detect objects in a picture. Multiple processing layers use image analysis filters, or convolutions as the model is trained.

The convolutional layers help extract the spatial features in an image. The layers have a weight-sharing technique, which helps in reducing computation efforts.

A Convolution Neural Network (CNN) is built on three broad strategies:

  1) Learn features using Convolution layer

  2) Reduce computational costs by down sample the image and reduce dimensionality using Max-Pooling (subsampling)

  3) Use fully connected layer to equip the network with classification capabilities

The performance of the model was evaluated with accuracy and recall scores. Accuracy is calculated by comparing the number correctly classified images (True Positives and True Negatives) to the total number of images. In contrast, recall compares scans correctly classified as having tumors (True Positives) to those misidentified as not having tumors (False Negatives). Given the dangers of misdiagnosing a patient with a tumor as being tumor-free, a high recall is desirable.

The original model incorporated a single convolutional layer and fully connected layer. This resulted in the results represented by the confusion matrix below, with an accuracy of 85% and a recall of 96%.  

![Initial Model Confusion Matrix](report/figures/confusion_matrix_fsm.png)

While the initial model had a promising recall score, there was plenty of room for improvement in accuracy, notice the number in the top right (False Positives) is greater than that in the top left (True Negatives). By adding additional epochs, the model showed improved in accuracy and recall, with a final accuracy and recall of 98% and 99%, respectively.

![Final Model Confusion Matrix](report/figures/confusion_matrix_final.png)


## LIME Visualization

To better understand how the model is classifying the images, I used LIME. LIME is an acronym for Local Interpretable Model-agnostic Explanations. The way LIME works is it breaks the image into several regions, called "superpixels", and the model classifies the image with the various superpixels turned on and off. Superpixels are then given "weights", a measure of importance for the model's classification. Below is a scan of a patient with a tumor, located in top region of the brain. Using LIME, we see with the superpixel with the highest weight is picking up on the region containing the tumor, evidence the model is identifying relevant regions when identifying the presence of tumors.

![LIME Visual](https://github.com/samjdedes/MRI_brain_scan_tumor_detection/blob/master/report/figures/LIME_True_Positive.png)


## Next Steps

With more data, next steps will be to classify images by location; whether is it in the frontal, parietal, temporal, occipital lobe or cerebellum, and whether the tumor is malignant or benign. Furthermore, there is a potential for early detection of tumors if labelled scans of undiagnosed brain tumors can be incorporated into the model.

There is potential to create separate models based on the type of section (frontal, horizontal, or medial), and this may show improvement in model accuracy and validity.


# Appendix
## Repository Navigation

Below visualizes the structure of the repository.

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

Technische Universität München. (n.d.). Planes of the Brain[Illustration]. Https://Wiki.Tum.De/. https://wiki.tum.de/download/attachments/29600620/Brain_directions_planes__sections_1_small.gif?version=1&modificationDate=1494257234627&api=v2

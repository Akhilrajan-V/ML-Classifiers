# ML-Classifiers for Face Recognition
### This Repository contains various Machine Learning Classifiers created from scratch such as Bayes, K-NN, Support Vector Machines, Boosted SVMs for face recognition on a custom dataset. Codes for datat preprocessing Principal Component Analysis(PCA) and Multiple Discriminant Analysis(MDA).

---
All classifiers are coded from scratch using only built-in python functions and numpy for matrix manipulation. 

Two classification tasks are carried out using various Machine Learning classifiers namely,

- Task 1: Subject/Label Recognition
- Task 2: Recognition of a Neutral vs an Expressive face  

The user can choose the classification task, data preprocessing method, and the classifier to perform classification from the terminal interface. 
The Bayes and KNN classifiers are applied to both the classification tasks. The SVM(Linear), Kernel-SVM, and Boosted SVMs are available only for Task 2. 

---
### REQUIREMENTS 

- python
- numpy
- matplotlib.pyplot 
- opencv
- sys


### DATA PREPROCESSING
The data is preprocessed before it is given to a classifier for either of the tasks. Dimensionality reduction algorithms to choose from,   
- Principal Component Analysis 
- Multiple Discriminant Analysis

**PRINCIPAL COMPONENT ANALYSIS**

The PCA preprocessing retains 99% of the original image variance while reducing the number of dimensions by more than half of that of the original image. 

To visualize PCA processed data, Run the `Preprocess/pca_visualizer.py` script. 
Change the data path  
```python
data_path = "/Your_absolute_path/Code/"
```
Left PCA Processed Image VS Right Original Image

![img](/Assets/pca_pic1.png)  ![img](/Assets/original_pic1.png) 

![img](/Assets/pca_pic2.png)  ![img](/Assets/original_pic2.png)

**MULTIPLE DISCRIMINANT ANALYSIS**

In the MDA script choose the number of components the data should be projected on. 

*NOTE Defalut is 200 components in the main script* 

[img](/Assets/mda_pic1.png)  ![img](/Assets/original_pic1.png) 

[img](/Assets/mda_pic2.png)  ![img](/Assets/original_pic2.png)

### TRAIN AND TEST DATA GENERATION
The preprocessed data is now split into train and test data. For Task 1: 2 images per subject are taken randomly for the train data while the remaining is used for test i.e., 1 image per label for test. For Task 2: All the neutral faces regardless of subject are given a common label and all the expressive face images are grouped together. Illumination images are discarded. `PLEASE See Data folder`.    

### TASK 1: Subject/Label Recognition 

The SVM classifiers are not available for this task. The Bayes classifier performs poorly due to high correlation among data. Bayes classifier uses MLE to estimate data mean and variance to model posterior.  

### TASK 2: Recognition of a Neutral vs an Expressive face 

All classifiers are available for this task. Note: Radial Basis Kernel SVM results in the highest classification accuracy at over 91%. 

### RUN THE CODE

Simply run the `Code/main.py` script from an IDE(recommended). Follow the terminal instructions to choose various available actions. 

**Before you run: Make sure to change data path in main.py, boosted_svm.py and pca_visualizer.py.**  :v:  

- [ ] Data.mat dataset implemented
- [x] Pose.mat dataset implemented
- [ ] MDA processing file optimized


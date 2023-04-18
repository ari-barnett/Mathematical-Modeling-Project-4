# Mathematical-Modeling-Project-4
# Barnett, Ari - University of Central Florida

Data available @ https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
This project entails taking a pretrained CNN and applying a siamese net approach to model access control for a company. 
An input image from the security checkpoint is evaluated via a distinct record file to determine access. 

To evaluate for deployablity of the approach at face value - multiple metrics were tested (euclidean distance, L1 distance, and Cosine
similarity measures). After individual examples where tested - a 1000 iteration test was run where a control and randomized test image where 
tried to determine access and TP,TN,FP,FN where recorded. Then specificty and sensitivity where calculated. 

Likely the model did not fare well due to features that where important to object recognition are not as important to facial features. 

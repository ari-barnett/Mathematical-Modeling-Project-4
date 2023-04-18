% Import and Restructure Net to output a feature vector to use 
net = resnet18("Weights","imagenet");

% Load Image Datasets and Hold as Augmented 
imds = imageDatastore('Celebrity Faces Dataset/', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
aug_imds = augmentedImageDatastore([224 224],imds,'ColorPreprocessing','gray2rgb');

feature_database = [];
net_application= classify(net,aug_imds);
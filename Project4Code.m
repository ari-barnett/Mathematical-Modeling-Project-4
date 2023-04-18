% Import and Restructure Net to output a feature vector to use 
net = layerGraph(resnet18("Weights","imagenet"));
featureNet = removeLayers(net, {'fc1000', 'prob','ClassificationLayer_predictions'});

% Create a dlnetwork object
featureNet = dlnetwork(featureNet);

% Load Image Datasets and Hold as Augmented 
imds = imageDatastore('Test Images/TEST IMDS/Cats/', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
aug_imds = augmentedImageDatastore([224 224],imds,'ColorPreprocessing','gray2rgb');

feature_database = zeros(height(aug_imds),512);

for i = 1:height(aug_imds.Files)
    disp(i);
    index = readByIndex(aug_imds,i);
    image_array = dlarray(double(cell2mat(index.input)),'SSCB');
    net_application= forward(featureNet,image_array);
    feature_vector = extractdata(reshape(net_application,[1 512]));

    feature_database(i,:) = feature_vector;
end

save('feature_database_Cats.mat',"feature_database");
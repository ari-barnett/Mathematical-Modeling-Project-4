Cat_imds = imageDatastore('Test Images/TEST IMDS/Cats/', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

Human_imds = imageDatastore('Celebrity Faces Dataset/', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%UNCOMMENT FOR CAT EXAMPLE
%load("feature_database_Cats.mat")
%f_matrix = feature_database(2,:);

load("feature_database.mat")
t_matrix = feature_database(1,:);
f_matrix = feature_database(1730,:); %Comment out for cat

figure()
subplot(1,2,1)
imshow(Human_imds.Files{1730})
%imshow(Cat_imds.Files{2}) %Uncomment for Cat

subplot(1,2,2)
imshow(Human_imds.Files{1})

A = f_matrix;
B = t_matrix;
AB = [A ; B];
reduction = tsne(AB,"Perplexity",0.5);

disp("======METRICS FOR FEATURE SPACE=======")
%Euclidean Distance for Vectors
distance = sqrt(sum((A - B).^2));
disp(distance)

%L1 Distance
distance = sum(abs(A - B));
disp(distance)

%Cosine Similarity
similarity = dot(A, B) / (norm(A) * norm(B));
disp(similarity)

%=================
disp("======METRICS FOR REDUCTION=======")
%Euclidean Distance for Vectors
distance = sqrt(sum((reduction(1,:) - reduction(2,:)).^2));
disp(distance)

%L1 Distance
distance = sum(abs(reduction(1,:) - reduction(2,:)));
disp(distance)

%Cosine Similarity
similarity = dot(reduction(1,:),reduction(2,:)) / (norm(reduction(1,:)) * norm(reduction(2,:)));
disp(similarity)


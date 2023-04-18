load("feature_database.mat")
warning('off', 'all');

%=========K means Clustering Example==============
correct_ID = 0;
staff_image = feature_database(1,:);

for i = 1:1000
    input_image = feature_database(randi([2 10]),:);
    evals = [input_image ; staff_image ; feature_database(randi([100 1800]),:)];
    [idx,C] = kmeans(evals,2);

    if idx(1) == idx(2)
        correct_ID = correct_ID + 1;
    end

end
a = (correct_ID/1000)*100;
%============================================


%========Dimension Reduction Distance Example======
staff_image = (feature_database(1,:));
control_test = (feature_database(1730,:));

TP = 0;
FP = 0;
TN = 0;
FN = 0;

threshold = 0.45;

sim_matrix = [];
control_matrix = [];
for i = 1:1000

    input_image =(feature_database(randi([2 10]),:));
    %evals = normalize([input_image ; staff_image],2,'range',[0 10]);
    %%UNCOMMENT FOR NORMALIZED DATA
    evals = ([input_image ; staff_image]);
    
    y = tsne(evals,Perplexity=0.5);
    
    similarity1 = dot(y(1,:), y(2,:)) / (norm(y(1,:)) * norm(y(2,:)));
    sim_matrix(i) = similarity1;
    
    %evals = normalize([control_test ; staff_image],2,'range',[0 10]);
    %%UNCOMMENT FOR NORMALIZED DATA
    evals = ([control_test ; staff_image]);
    
    y = tsne(evals,Perplexity=0.5);
    similarity2 = dot(y(1,:), y(2,:)) / (norm(y(1,:)) * norm(y(2,:)));
    control_matrix(i) = similarity2;


    if similarity1 > threshold
        TP = TP + 1;
    elseif similarity1 < threshold
        FN = FN + 1;
    end

    if similarity2 < threshold
        TN = TN + 1;
    elseif similarity2 > threshold
        FP = FP + 1;
    end
   
end

disp("KMEAN APPROACH")
disp("Correctly identified employee = " + a + "% of the time")
disp("   ")


disp("COSINE SIM. APPROACH")
Sensitiviy = TP / (TP + FN);
Specificty = TN / (TN + FP);
Accurary = (TP + TN)/(TP+TN+FP+FN); 
disp("Specifity = " + Specificty + "  Sensitivty = " + Sensitiviy);
disp("Accuracy = "+ Accurary);
disp("  ");

disp("MEDIAN COSINE SIMILARITY")
disp("Similarity Average (True) = " + median(sim_matrix));
disp("Similarity Average (False) = " + median(control_matrix));
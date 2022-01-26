%%Prepare data
clear;clc

load('5FoldCV.mat');

accuracy = [];
predict_labels = zeros(600,1);
scores = zeros(600,1);
%%Configure Net
net = feedforwardnet(10, 'traingd');

net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
net.inputs{1}.processFcns = {}; % modify the process function for inputs
net.outputs{2}.processFcns = {}; % modify the process function for outputs
net.layers{1}.transferFcn = 'poslin'; % the transfer function for the first layer: ReLU
net.layers{2}.transferFcn = 'softmax'; % the transfer function for the second layer
net.performFcn = 'crossentropy'; % loss function
%%Adjust Training Parameters
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1; % learning rate.

%%5-Fold Validation
for i = 1:5
    trainIdx = cvp.training(i); %% get the index of training samples
    testIdx = cvp.test(i); %% get the index of the test samples
    training_label = class(trainIdx); %% create the training label ground truth
    training_instance = imgs(trainIdx,:); %% create the training data
    test_label = class(testIdx); %% create the testing label ground truth
    test_instance = imgs(testIdx,:); %% create the test data
    
    %%Training
    model = train(net, training_instance.', training_label.');
    
    %%Test
    scores(testIdx,:) = sim(model, test_instance.');
    
    %%Obtain Labels
    thresh = 0.5;
    predict_labels(scores<thresh,:) = 0;
    predict_labels(scores>=thresh,:) = 1;
    
    %%Compute Accuracy
    correct_count(:,i) = (sum(predict_labels(testIdx,:)==class(testIdx)));
    accuracy(:,i) = correct_count(:,i) ./ size(test_label);
end

%%Plot Figures
figure(1)
accuracy_avg = mean(accuracy(1,:));
confusionchart(class, predict_labels)
Title = sprintf('Confusion Matrix of NN (Acc = %.2f%%)',accuracy_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')

figure(2)
[FPR, TPR, AUC] = roccurv(class, scores);
fill([FPR 0 1 1],[TPR 0 0 1],[0.529, 0.808, 0.922]); hold on
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold off
title('ROC Curve of NN')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
box on

figure(3)
plot(FPR,TPR,'linewidth',2); hold on
%%
save classify_results/nn_result
%%
%%Prepare data
clear;clc

load('5FoldCV.mat');

accuracy = [];
predict_labels = zeros(600,1);
scores = zeros(600,1);
%%Configure Net
net = feedforwardnet([5,5], 'traingd');

net.divideParam.trainRatio = 1; % training set [%]
net.divideParam.valRatio = 0; % validation set [%]
net.divideParam.testRatio = 0; % test set [%]
net.inputs{1}.processFcns = {}; % modify the process function for inputs
net.outputs{2}.processFcns = {}; % modify the process function for outputs
net.layers{1}.transferFcn = 'poslin'; % the transfer function for the first layer: ReLU
net.layers{2}.transferFcn = 'poslin'; % the transfer function for the second layer: ReLU
net.layers{3}.transferFcn = 'softmax'; % the transfer function for the output layer
net.performFcn = 'crossentropy'; % loss function
%%Adjust Training Parameters
net.trainParam.epochs = 10000;
net.trainParam.lr = 0.1; % learning rate.

%%5-Fold Validation
for i = 1:5
    trainIdx = cvp.training(i); %% get the index of training samples
    testIdx = cvp.test(i); %% get the index of the test samples
    training_label = class(trainIdx); %% create the training label ground truth
    training_instance = imgs(trainIdx,:); %% create the training data
    test_label = class(testIdx); %% create the testing label ground truth
    test_instance = imgs(testIdx,:); %% create the test data
    
    %%Training
    model = train(net, training_instance.', training_label.');
    
    %%Test
    scores(testIdx,:) = sim(model, test_instance.');
    
    %%Obtain Labels
    thresh = 0.55;
    predict_labels(scores<thresh,:) = 0;
    predict_labels(scores>=thresh,:) = 1;
    
    %%Compute Accuracy
    correct_count(:,i) = (sum(predict_labels(testIdx,:)==class(testIdx)));
    accuracy(:,i) = correct_count(:,i) ./ size(test_label);
end

%%Plot Figures
figure(1)
accuracy_avg = mean(accuracy(1,:));
confusionchart(class, predict_labels)
Title = sprintf('Confusion Matrix of NN (Thresh = 0.55, Acc = %.2f%%)',accuracy_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')

figure(2)
[FPR, TPR, AUC] = roccurv(class, scores);
fill([FPR 0 1 1],[TPR 0 0 1],[0.529, 0.808, 0.922]); hold on
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold off
title('ROC Curve of NN')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
box on
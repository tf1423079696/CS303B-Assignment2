%% Linear Kernel
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using a linear kernel
%%Do 5-Fold Cross Validation
model = fitcsvm(imgs, class, 'KernelFunction','linear','CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
% [FPR, TPR, Th, AUC] = perfcurve(class, score(:,2),'1');
figure(2)
fill([FPR 0 1 1],[TPR 0 0 1],[0.529, 0.808, 0.922]); hold on
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold off
title('ROC Curve of Linear SVM')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
box on

figure(3)
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold on
title('ROC Curves of Three Classifiers')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
box on
grid on


%%Compute Accuracy
loss = [];
accur = [];
loss(1) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 3 4]);accur(1) = 1 - loss(1);
loss(2) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 3 5]);accur(2) = 1 - loss(2);
loss(3) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 4 5]);accur(3) = 1 - loss(3);
loss(4) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 3 4 5]);accur(4) = 1 - loss(4);
loss(5) = kfoldLoss(model,'LossFun','classiferror','Folds',[2 3 4 5]);accur(5) = 1 - loss(5);
loss_avg = mean(loss)
accur_avg = mean(accur)

figure(1)
confusionchart(class, predict_label)
Title = sprintf('Confusion Matrix of Linear SVM (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')

save classify_results/svm_linear_result
%% RBF Kernel
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.01;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
% [FPR, TPR, Th, AUC] = perfcurve(class, score(:,2),'1');
fill([FPR 0 1 1],[TPR 0 0 1],[0.529, 0.808, 0.922]); hold on
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold off
title('ROC Curve of SVM with RBF Kernel')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
box on

figure(3)
plot(FPR,TPR,'linewidth',2); hold on

%%Compute Accuracy
loss = [];
accur = [];
loss(1) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 3 4]);accur(1) = 1 - loss(1);
loss(2) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 3 5]);accur(2) = 1 - loss(2);
loss(3) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 2 4 5]);accur(3) = 1 - loss(3);
loss(4) = kfoldLoss(model,'LossFun','classiferror','Folds',[1 3 4 5]);accur(4) = 1 - loss(4);
loss(5) = kfoldLoss(model,'LossFun','classiferror','Folds',[2 3 4 5]);accur(5) = 1 - loss(5);
loss_avg = mean(loss)
accur_avg = mean(accur)

figure(1)
confusionchart(class, predict_label)
Title = sprintf('Confusion Matrix of SVM with RBF Kernel (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')

save classify_results/svm_rbf_result
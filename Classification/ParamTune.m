%% RBF Kernel with gamma = 1000
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 1000;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold on
title('ROC Curves of SVM with RBF Kernel with different \gamma')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
grid on
box on

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
save param_results/1000

%% RBF Kernel with gamma = 10
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 10;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
save param_results/10

%% RBF Kernel with gamma = 1
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 1;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
colormap(summer)
plot(linspace(0,1,600),linspace(0,1,600),'--k','linewidth',0.5);hold on
plot(FPR,TPR,'linewidth',2); hold on
title('ROC Curves of SVM with RBF Kernel with different \gamma')
xlabel('false positive rate')
ylabel('true positive rate')
xlim([0,1]);ylim([0,1])
grid on
box on

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
Title = sprintf('Confusion Matrix of gamma=1 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/1
%% RBF Kernel with gamma = 0.1
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.1;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.1 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/01

%% RBF Kernel with gamma = 0.08
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.08;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.08 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/008
%% RBF Kernel with gamma = 0.05
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.05;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.05 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/005
%% RBF Kernel with gamma = 0.03
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.03;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.03 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/003

%% RBF Kernel with gamma = 0.01
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
Title = sprintf('Confusion Matrix of gamma=0.01 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/001
%% RBF Kernel with gamma = 0.005
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.005;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.005 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/0005
%% RBF Kernel with gamma = 0.001
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.001;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.001 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/0001
%% RBF Kernel with gamma = 0.0005
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.0005;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.0005 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/00005

%% RBF Kernel with gamma = 0.0001
clear;clc

%%Prepare data
load('5FoldCV.mat');

%%Train SVM using RBF kernel
%%Do 5-Fold Cross Validation
gamma = 0.0001;
model = fitcsvm(imgs, class, 'KernelFunction','rbf','KernelScale',1/sqrt(gamma),'CVPartition',cvp);
model.ScoreTransform = 'logit';

%%ROC & AUC
[predict_label,score] = kfoldPredict(model);
[FPR, TPR, AUC] = roccurv(class, predict_label);
figure(2)
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
Title = sprintf('Confusion Matrix of gamma=0.0001 (Acc = %.2f%%)',accur_avg*100);
sgtitle(Title);
xlabel('Prediction');ylabel('Ground Truth')
save param_results/00001

%%
figure(2)
colormap(summer)
r = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 1, 10, 1000];
acc = [0.6883, 0.9, 0.9417, 0.9750, 0.9817, 0.985, 0.9817, 0.95, 0.8567, 0.6883, 0.6883, 0.6883];
auc = [0.5203, 0.8376, 0.9192, 0.9695, 0.9795, 0.9826, 0.9765, 0.92, 0.7923, 0.5203, 0.5203, 0.5203];
x = log(r);
plot(x, acc, 'linewidth',2);hold on
plot(x, auc, 'linewidth',2);hold off
xlabel('log(\gamma)')
ylabel('performance')
title('Variation Trend of Accuracy and AUC with \gamma')
grid on
legend('accuracy','AUC')
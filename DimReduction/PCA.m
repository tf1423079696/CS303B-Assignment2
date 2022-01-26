clear; clc

%%Load Data
%%images:
%%784 rows: 784 pixels
%%600 columns: 600 images
load('../mnist-1-5-8.mat');

%%将图片按1，5，8分类
digit_one = []; count_one = 0;
digit_five = []; count_five = 0;
digit_eight = []; count_eight = 0;
for i=1:600
    if(labels(i) == 1)
        count_one = count_one+1;
        digit_one(:, count_one) = images(:, i);
    elseif(labels(i) == 5)
        count_five = count_five+1;
        digit_five(:, count_five) = images(:, i);
    elseif(labels(i) == 8)        
        count_eight = count_eight+1;
        digit_eight(:, count_eight) = images(:, i);
    end
end

%%Normalize
imgs = images;
for i=1:size(images,1)
    data = images(i,:);
%     data = (data-min(data(:)))./(max(data(:))-min(data(:))+0.001);
    [data,~] = mapminmax(data,0,1);
    imgs(i,:) = data;
end

%%Compute Mean Value Matrix
E = mean(imgs,2);
E = repmat(E, 1, size(images, 2));

%%Decentralize
imgs = imgs - E;

%%Compute Covariance Matrix
C = cov(imgs.');
%C = (1/size(images, 2)-1) * (imgs * imgs');

%%Compute Eigenvectors and Eigenvalues, i.e., principal components
[eigvcts, eigvals] = eigs(C);

%%Find the First Two Principal Components
eigmaxs = max(abs(eigvals));
[prinpval_1, prinploc_1] = max(eigmaxs);
eigmaxs(prinploc_1) = 0;
[prinpval_2, prinploc_2] = max(eigmaxs);
prinpvct_1 = eigvcts(:,prinploc_1);
prinpvct_2 = eigvcts(:,prinploc_2);

%%Compute Projections to the First Two Principal Components
proj_1 = prinpvct_1.' * imgs;
proj_2 = prinpvct_2.' * imgs;

figure(1)
colormap(summer)
scatter(proj_1(:,labels==1), proj_2(:,labels==1),20,'o');hold on
scatter(proj_1(:,labels==5), proj_2(:,labels==5),'x');hold on
scatter(proj_1(:,labels==8), proj_2(:,labels==8),'*');hold off
box on
legend('1','3','5')
xlabel('projection on 1st principle component')
ylabel('projection on 2nd principle component')
title('Dimension Reduction with PCA')

figure(2)
%%Randomly choose two features of original data
a = unidrnd(size(images,2))
b = unidrnd(size(images,2))
colormap(summer)
scatter(imgs(a,labels==1), imgs(b,labels==1),20,'o');hold on
scatter(imgs(a,labels==5), imgs(b,labels==5),'x');hold on
scatter(imgs(a,labels==8), imgs(b,labels==8),'*');hold off
box on
legend('1','3','5')
x_label = sprintf('%dth feature',a);
y_label = sprintf('%dth feature',b);
xlabel(x_label)
ylabel(y_label)
title('Scatter of Features Before Dim-reduction')

save ../Data/data_PCA proj_1 proj_2 labels
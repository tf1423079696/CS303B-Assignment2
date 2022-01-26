clear; clc

%%Load Data
%%images:
%%784 rows: 784 pixels
%%600 columns: 600 images
load('../mnist-1-5-8.mat');

%%Normalize
for i=1:size(images,1)
    data = images(i,:);
    [data,~] = mapminmax(data,0,1);
    images(i,:) = data;
end

%%Classify Images According to '1'，'5'，'8'
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

%%Compute Average of all Classes
C = mean(images,2);
%%Compute Average of Each Class
C1 = mean(digit_one,2);
C5 = mean(digit_five,2);
C8 = mean(digit_eight,2);

%%Compute between-class scatter
Mb = [C1-C C5-C C8-C];
Sb = Mb*(Mb.');

%%Compute between-class scatter
Mw = [digit_one-C1 digit_five-C5 digit_eight-C8];
Sw = Mw*(Mw.');

%%Solve eigenvalue problem of inv(Sw)*Sb
I = eye(size(Sw));
[eigvcts, eigvals] = eigs((Sw+0.0001*I)\Sb);

%%Find w1 & w2
eigmaxs = max(abs(eigvals));
[~, wloc_1] = max(eigmaxs);
eigmaxs(wloc_1) = 0;
[~, wloc_2] = max(eigmaxs);
w_1 = eigvcts(:,wloc_1);
w_2 = eigvcts(:,wloc_2);

%%Compute Projections to w1 & w2
E = repmat(C, 1, size(images, 2));
imgs = images - E;
proj_1 = w_1.' * imgs;
proj_2 = w_2.' * imgs;

figure(1)
colormap(summer)
scatter(proj_1(:,labels==1), proj_2(:,labels==1),20,'o');hold on
scatter(proj_1(:,labels==5), proj_2(:,labels==5),'x');hold on
scatter(proj_1(:,labels==8), proj_2(:,labels==8),'*');hold off
box on
legend('1','3','5')
xlabel('')
ylabel('')
title('Dimension Reduction with LDA')
save ../Data/data_LDA proj_1 proj_2 labels
%% Cluster after PCA
clear;clc
load ../Data/data_PCA
data = [reshape(proj_1,[600,1]) reshape(proj_2,[600,1])]; clear proj_1 proj_2
posit_one = find(labels==1);posit_five = find(labels==5);posit_eight = find(labels==8);

%%Randomly Set Initial Centers 
clust1 = [];clust2 = [];clust3 = [];
centr = [data(unidrnd(size(data,1)),:); data(unidrnd(size(data,1)),:); data(unidrnd(size(data,1)),:)];
centr_new = [[]; []; []];

dist = pdist2(centr, data);
[~,class] = min(dist);
clust1 = data(class==1,:); clust2 = data(class==2,:); clust3 = data(class==3,:);
centr_new = [mean(clust1); mean(clust2); mean(clust3)];

%%Iteration
while(~isequal(centr, centr_new))
    centr = centr_new;
    dist = pdist2(centr, data,'cosine');
    [~,class] = min(dist);
    clust1 = data(class==1,:); clust2 = data(class==2,:); clust3 = data(class==3,:);
    centr_new = [mean(clust1); mean(clust2); mean(clust3)];
end

confus = zeros(3,3);
for i = 1:size(labels)
    if(class(:,i)==1 && labels(i,:)==1)
        confus(1,1) = confus(1,1) + 1;
    elseif(class(:,i)==1 && labels(i,:)==5)
        confus(2,1) = confus(2,1) + 1;
    elseif(class(:,i)==1 && labels(i,:)==8)
        confus(3,1) = confus(3,1) + 1;
    elseif(class(:,i)==2 && labels(i,:)==1)
        confus(1,2) = confus(1,2) + 1;
    elseif(class(:,i)==2 && labels(i,:)==5)
        confus(2,2) = confus(2,2) + 1;
    elseif(class(:,i)==2 && labels(i,:)==8)
        confus(3,2) = confus(3,2) + 1;
    elseif(class(:,i)==3 && labels(i,:)==1)
        confus(1,3) = confus(1,3) + 1;
    elseif(class(:,i)==3 && labels(i,:)==5)
        confus(2,3) = confus(2,3) + 1;
    elseif(class(:,i)==3 && labels(i,:)==8)
        confus(3,3) = confus(3,3) + 1;
    end
end

accur = 0;
max_post = [];
for i=1:3
    [correct_num, max_post(i)] = max(confus(i,:));
    accur = accur + correct_num;
end
accur = accur ./ size(labels)


figure(1)
colormap(summer)
scatter(clust1(:,1), clust1(:,2),20,'o');hold on
scatter(clust2(:,1), clust2(:,2),'x');hold on
scatter(clust3(:,1), clust3(:,2),'*');hold on
scatter(centr(1,1), centr(1,2),'k','d','filled');hold on
scatter(centr(2,1), centr(2,2),'k','d','filled');hold on
scatter(centr(3,1), centr(3,2),'k','d','filled');hold off
box on
legend('cluster 1','cluster 2','cluster 3','centers')
str = sprintf('Cluster Results (Accuracy=%.2f%%)',accur(:,1)*100);
title(str)

figure(2)
colormap(summer)
scatter(data(labels==1,1), data(labels==1,2),20,'o');hold on
scatter(data(labels==5,1), data(labels==5,2),'x');hold on
scatter(data(labels==8,1), data(labels==8,2),'*');hold off
box on
legend('1','3','5')
title('Ground Truth')
%% Cluster after LDA
clear;clc
load ../Data/data_LDA
data = [reshape(proj_1,[600,1]) reshape(proj_2,[600,1])]; clear proj_1 proj_2

clust1 = [];clust2 = [];clust3 = [];
%centr = [data(2,:); data(1,:); data(7,:)];
centr = [data(unidrnd(size(data,1)),:); data(unidrnd(size(data,1)),:); data(unidrnd(size(data,1)),:)];
centr_new = [[]; []; []];

dist = pdist2(centr, data);
[~,class] = min(dist);
clust1 = data(class==1,:); clust2 = data(class==2,:); clust3 = data(class==3,:);
centr_new = [mean(clust1); mean(clust2); mean(clust3)];

while(~isequal(centr, centr_new))
    centr = centr_new;
    dist = pdist2(centr, data);
    [~,class] = min(dist);
    clust1 = data(class==1,:); clust2 = data(class==2,:); clust3 = data(class==3,:);
    centr_new = [mean(clust1); mean(clust2); mean(clust3)];
end

confus = zeros(3,3);
for i = 1:size(labels)
    if(class(:,i)==1 && labels(i,:)==1)
        confus(1,1) = confus(1,1) + 1;
    elseif(class(:,i)==1 && labels(i,:)==5)
        confus(2,1) = confus(2,1) + 1;
    elseif(class(:,i)==1 && labels(i,:)==8)
        confus(3,1) = confus(3,1) + 1;
    elseif(class(:,i)==2 && labels(i,:)==1)
        confus(1,2) = confus(1,2) + 1;
    elseif(class(:,i)==2 && labels(i,:)==5)
        confus(2,2) = confus(2,2) + 1;
    elseif(class(:,i)==2 && labels(i,:)==8)
        confus(3,2) = confus(3,2) + 1;
    elseif(class(:,i)==3 && labels(i,:)==1)
        confus(1,3) = confus(1,3) + 1;
    elseif(class(:,i)==3 && labels(i,:)==5)
        confus(2,3) = confus(2,3) + 1;
    elseif(class(:,i)==3 && labels(i,:)==8)
        confus(3,3) = confus(3,3) + 1;
    end
end

accur = 0;
max_post = [];
for i=1:3
    [correct_num, max_post(i)] = max(confus(i,:));
    accur = accur + correct_num;
end
accur = accur ./ size(labels)


figure(1)
colormap(summer)
scatter(clust1(:,1), clust1(:,2),20,'o');hold on
scatter(clust2(:,1), clust2(:,2),'x');hold on
scatter(clust3(:,1), clust3(:,2),'*');hold on
scatter(centr(1,1), centr(1,2),'k','d','filled');hold on
scatter(centr(2,1), centr(2,2),'k','d','filled');hold on
scatter(centr(3,1), centr(3,2),'k','d','filled');hold off
box on
legend('cluster 1','cluster 2','cluster 3','centers')
str = sprintf('Cluster Results (Accuracy=%.2f%%)',accur(:,1)*100);
title(str)

figure(2)
colormap(summer)
scatter(data(labels==1,1), data(labels==1,2),20,'o');hold on
scatter(data(labels==5,1), data(labels==5,2),'x');hold on
scatter(data(labels==8,1), data(labels==8,2),'*');hold off
box on
legend('1','3','5')
title('Ground Truth')
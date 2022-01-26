%% Cluster after PCA
clear;clc
load ../Data/data_PCA
data = [reshape(proj_1,[600,1]) reshape(proj_2,[600,1])]; clear proj_1 proj_2
[length,~] = size(data);
clust = [];

%%Write Hierarchical Clustering Codes Myself
% Dist = pdist2(data,data);
% dist = Dist;
% for i = 1:length
%     for j = i:length
%         dist(j,i) = Inf;
%     end
%     Dist(i,i) = Inf;
% end
%
% clust_num = length;
% while(~(clust_num == 3 || clust_num < 3))
%     [r,c] = find(dist == min(min(dist)));
%     r = r(1);c=c(1);
%     if(ismember(r,clust))
%         [label,~] = find(clust == r);
%         clust(label,end+1) = c;
%     elseif(ismember(c,clust))
%         [label,~] = find(clust == c);
%         clust(label,end+1) =  r;
%     else
%         clust = [clust;r c];
%     end
%     
%     [label,~] = find(clust == r)
%     
%     inter_Dist = Dist(clust(label,1),:);
%     for i = 2:size(clust(label,:),2)
%         inter_Dist = inter_Dist + Dist(clust(label,i),:);
%     end
%     inter_dist = inter_Dist ./ size(clust(label,:),2);
%     
%     for i = 1:size(clust(label,:),2)
%         Dist(clust(label,i),:) = inter_dist;
%         Dist(:,clust(label,i)) = inter_dist.';
%     end
%     
%     dist = Dist;
%     for i = 1:length
%         for j = i:length
%             dist(j,i) = Inf;
%         end
%     end
%     clust_num = clust_num-1;
% end

dist = pdist(data,'cosine');
%%Use Average Linkage
process = linkage(dist,'average');
correlation = cophenet(process,dist)

%%Expected Cluster Number = 3
clust = cluster(process,'maxclust',3);

confus = zeros(3,3);
for i = 1:size(labels)
    if(clust(i,:)==1 && labels(i,:)==1)
        confus(1,1) = confus(1,1) + 1;
    elseif(clust(i,:)==1 && labels(i,:)==5)
        confus(2,1) = confus(2,1) + 1;
    elseif(clust(i,:)==1 && labels(i,:)==8)
        confus(3,1) = confus(3,1) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==1)
        confus(1,2) = confus(1,2) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==5)
        confus(2,2) = confus(2,2) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==8)
        confus(3,2) = confus(3,2) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==1)
        confus(1,3) = confus(1,3) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==5)
        confus(2,3) = confus(2,3) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==8)
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

figure(1);
scatter(data(clust==1,1), data(clust==1,2),20,'o');hold on
scatter(data(clust==2,1), data(clust==2,2),'x');hold on
scatter(data(clust==3,1), data(clust==3,2),'*');hold off
box on
legend('cluster 1','cluster 2','cluster 3')
str = sprintf('Hierarchical Clustering Results (Accuracy=%.2f%%)',accur(:,1)*100);
title(str)

figure(2);
colormap(summer)
scatter(data(labels==1,1), data(labels==1,2),20,'o');hold on
scatter(data(labels==5,1), data(labels==5,2),'x');hold on
scatter(data(labels==8,1), data(labels==8,2),'*');hold off
box on
legend('1','5','8')
title('Ground Truth')

figure(3)
D = dendrogram(process,'ColorThreshold','default');
set(D,'LineWidth',1.5)
box on
title('Process of Hierarchical Clustering')
%% Cluster after LDA
clear;clc
load ../Data/data_LDA
data = [reshape(proj_1,[600,1]) reshape(proj_2,[600,1])]; clear proj_1 proj_2
clust = [];

dist = pdist(data);
%%Use Average Linkage
process = linkage(dist,'average');
correlation = cophenet(process,dist)

%%Expected Cluster Number = 3
clust = cluster(process,'maxclust',3);

figure(1);
scatter(data(clust==1,1), data(clust==1,2),'r');hold on
scatter(data(clust==2,1), data(clust==2,2),'g');hold on
scatter(data(clust==3,1), data(clust==3,2),'b');hold off
box on
legend('cluster 1','cluster 2','cluster 3')
title('Cluster Results')

figure(2);
scatter(data(labels==1,1), data(labels==1,2),'r');hold on
scatter(data(labels==5,1), data(labels==5,2),'g');hold on
scatter(data(labels==8,1), data(labels==8,2),'b');hold off
box on
legend('1','5','8')
title('Ground Truth')

confus = zeros(3,3);
for i = 1:size(labels)
    if(clust(i,:)==1 && labels(i,:)==1)
        confus(1,1) = confus(1,1) + 1;
    elseif(clust(i,:)==1 && labels(i,:)==5)
        confus(2,1) = confus(2,1) + 1;
    elseif(clust(i,:)==1 && labels(i,:)==8)
        confus(3,1) = confus(3,1) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==1)
        confus(1,2) = confus(1,2) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==5)
        confus(2,2) = confus(2,2) + 1;
    elseif(clust(i,:)==2 && labels(i,:)==8)
        confus(3,2) = confus(3,2) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==1)
        confus(1,3) = confus(1,3) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==5)
        confus(2,3) = confus(2,3) + 1;
    elseif(clust(i,:)==3 && labels(i,:)==8)
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

figure(1);
scatter(data(clust==1,1), data(clust==1,2),20,'o');hold on
scatter(data(clust==2,1), data(clust==2,2),'x');hold on
scatter(data(clust==3,1), data(clust==3,2),'*');hold off
box on
legend('cluster 1','cluster 2','cluster 3')
str = sprintf('Hierarchical Clustering Results (Accuracy=%.2f%%)',accur(:,1)*100);
title(str)

figure(2);
colormap(summer)
scatter(data(labels==1,1), data(labels==1,2),20,'o');hold on
scatter(data(labels==5,1), data(labels==5,2),'x');hold on
scatter(data(labels==8,1), data(labels==8,2),'*');hold off
box on
legend('1','5','8')
title('Ground Truth')
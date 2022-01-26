clear;clc

load('../mnist-1-5-8.mat');
[length,~] = size(images);
imgs = images.';
for i=1:length
    data = images(i,:);
    data = (data-min(data(:)))./(max(data(:))-min(data(:))+0.001);
    imgs(:,i) = data;
end

%%Prepare the vectors for '5' against the rest
class = zeros(size(labels));
for i=1:size(class)
    class(i) = (labels(i)==5);
    %1 means '5'; '0' means others
end

cvp = cvpartition(class,'KFold',5);

save 5FoldCV cvp class imgs images labels
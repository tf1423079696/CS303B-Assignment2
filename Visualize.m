clear; clc

%%Load Data
%%images:
%%784 rows: 784 pixels
%%600 columns: 600 images
load('mnist-1-5-8.mat');

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
for i= 1:3
    im = reshape(digit_one(:,9*i), [28, 28]); % image of the digit '1'
    name = sprintf('Images/1_%d.jpg',i)
    imwrite(im, name);

    im = reshape(digit_five(:,9*i), [28, 28]); % image of the digit '5'
    name = sprintf('Images/5_%d.jpg',i)
    imwrite(im, name);

    im = reshape(digit_eight(:,9*i), [28, 28]); % image of the digit '8'
    name = sprintf('Images/8_%d.jpg',i)
    imwrite(im, name);
end
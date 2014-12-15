% CNS 187 - Sparse Code for Natural Images
% Author: James Chang

%% Parameters
clear all;
param.sampleSize = [8,8];
param.numPixels = param.sampleSize(1) * param.sampleSize(2);
param.filename = 'Image Data.mat';
param.DEBUG = 1;
results = {};
param.R = [2, 8, 32];

%% 1. Select a natural image with resolution of at least 640x480
%{
image = rgb2gray(imread('image.jpeg'));
results.image = image;
%}

% Or load from the mnist data set!
load('mnist_train.mat')

count = 1;
result = [];
for row = 1:23
    row = [];
    for col = 1:23
        image = data(:,:,count);
        row = [row image];
        count = count + 1;
    end
    result = [result; row];
end
image = result;

% Display the image
figure;
colormap gray;
imagesc(image);
title('Figure 3: Original Image');

%% 2. Sample 8x8 patches from the image and accumulate them as column vectors
A = sampleImage(image, param);
results.A = A;

%% 3. Use SVD on the sampled matrix to compute the principal components
[U,S,V] = svd(A, 'econ');
results.U = U;
results.S = S;
results.V = V;

%% 4. Plot the singular values in a loglog plot of the diagonals of S
singulars = zeros(1, param.sampleSize(1) * param.sampleSize(2));

for i = 1:(param.sampleSize(1) * param.sampleSize(2))
    singulars(i) = S(i,i);
end

results.singulars = singulars;


figure;
loglog(singulars);
title('Figure 1: SVD Singular Values');
xlabel('Column Number');
ylabel('Singular Value');
grid on;


%% 5. Display the first 64 columns of U as images

figure;
colormap gray;
for i = 1:64
    subplot(8, 8, i);
    imagesc(mat2gray(reshape(U(:,i), 8, 8)));
    title(i)  
end
ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off',...
    'Visible','off','Units','normalized', 'clipping' , 'off');
text(0.5, 1,'Figure 2: First 64 Images of SVD U','HorizontalAlignment',...
    'center','VerticalAlignment', 'top');


%% 6. Reconstruct the original image with the first R components
% 3 values of R
[reconstructs errors] = reconstructImage(U, S, V, image, A, param);
results.reconstructs = reconstructs;
results.errors = errors;

% Plot our reconstructions
for i = 1:numel(reconstructs)
    figure;
    colormap gray;
    imagesc(mat2gray(reconstructs{1}));
    title(['Figure ', num2str(i+3), ': Reconstruction with R = ', ...
        num2str(param.R(i))]);
end

%% 7. Errors
errors

%% 8. Save Results
save(param.filename, 'results')
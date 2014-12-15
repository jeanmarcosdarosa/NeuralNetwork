clear all;

%% LOAD DATASET AND SET PARAMETERS

% Dimensions of all layers (except input)
paramStruct.layersDims = [3,1];
% Number of layers
paramStruct.numLayers = length(paramStruct.layersDims) + 1;
% Learning rate of the gradient descent weight update
paramStruct.learningRate = 0.01;
% Maximum number of iterations of the backpropagation
paramStruct.maxIterations = 1000;
% The categories we are classifying
paramStruct.labelTypes = [2, 7];
% Normalized categories we are classifying
paramStruct.nLabelTypes = mat2gray(paramStruct.labelTypes);
% How many categories we are classifying
paramStruct.numLabelTypes = numel(paramStruct.labelTypes);
% Gamma for regularization
paramStruct.gamma = 0.01;
% L1 on or off
paramStruct.L1 = 0;
% L2 on or off
paramStruct.L2 = 0;
% Track weights
paramStruct.trackWeights = 0;
paramStruct.trackErrors = 0;
paramStruct.normalizeYs = 1;


%% Initial training for 2s and 7s
%{
load('mnist_train.mat');
[Xtr, ytr] = parseData(data, labels, 'Training-2-7.mat', paramStruct);
load('mnist_test.mat')
[Xts, yts] = parseData(data, labels, 'Test-2-7.mat', paramStruct);
%}


%% Training regular vs L1 vs L2
%{
paramStruct.layersDims = [3,1];
trainThenTest(Xtr, ytr, Xts, yts, '1500, 2-7-[3-1].mat', paramStruct);
paramStruct.L1 = 1;
trainThenTest(Xtr, ytr, Xts, yts, '1500L1, 2-7-[3-1].mat', paramStruct);
paramStruct.L1 = 0;
paramStruct.L2 = 1;
trainThenTest(Xtr, ytr, Xts, yts, '1500L2, 2-7-[3-1].mat', paramStruct);
%}

%% Training for 3 digit case
%{
load('Training-2-7.mat');
Xtr = input_data.X;
ytr = input_data.y;
load('Test-2-7.mat');
Xts = input_data.X;
yts = input_data.y;
paramStruct.labelTypes = [2, 7];
trainThenTest(Xtr, ytr, Xts, yts, '1000, 2-7-[3-1].mat', paramStruct);

paramStruct.labelTypes = [2, 9];
load('mnist_train.mat');
[Xtr, ytr] = parseData(data, labels, 'Training-2-9.mat', paramStruct);
load('mnist_test.mat')
[Xts, yts] = parseData(data, labels, 'Test-2-9.mat', paramStruct);
trainThenTest(Xtr, ytr, Xts, yts, '1000, 2-9-[3-1].mat', paramStruct);

paramStruct.labelTypes = [7, 9];
load('mnist_train.mat');
[Xtr, ytr] = parseData(data, labels, 'Training-7-9.mat', paramStruct);
load('mnist_test.mat')
[Xts, yts] = parseData(data, labels, 'Test-7-9.mat', paramStruct);
trainThenTest(Xtr, ytr, Xts, yts, '1000, 7-9-[3-1].mat', paramStruct);
%}

%% PLOTTING ERRORS for no regularization, L1, L2
%{
load('1500, 2-7-[3-1].mat');
iter = 1:(paramStruct.maxIterations + 1);

figure;
loglog(iter, results.trainingErrors, 'r');
title('Figure 1:Training Error with no Regularization');
xlabel('Iteration');
ylabel('Error');
grid on;

load('1500L1, 2-7-[3-1].mat');
figure;
loglog(iter, results.trainingErrors, 'g');
title('Figure 2: Training Error with L1 Regularization');
xlabel('Iteration');
ylabel('Error');
grid on;

load('1500L2, 2-7-[3-1].mat');
figure;
loglog(iter, results.trainingErrors, 'b');
title('Figure 3: Training Error with L2 Regularization');
xlabel('Iteration');
ylabel('Error');
grid on;
%}

%% PLOTTING WEIGHT changes over iterations for no regularization, L1, and L2
%{
figure;

load('1500L1, 2-7-[3-1].mat');

w = results.trainingWeightList;
toPlot = cell(1, numel(w{1}{1}) + numel(w{1}{2}));
for i = 1:numel(toPlot)
    toPlot{i} = zeros(1, paramStruct.maxIterations);
end

for i = 1:paramStruct.maxIterations
    count = 1;
    for weight = 1:numel(w{i})
        for j = 1:numel(w{i}{weight})
            toPlot{count}(i) = w{i}{weight}(j);
            count = count + 1;
        end
    end
end

for i = 1:numel(toPlot)
    plot(1:1500, mat2gray(toPlot{i}));
    hold on;
end
title('Figure 4: Weight Transformations with L1 Regularization');
xlabel('Iteration');
ylabel('Weight Values');
grid on;
hold off;
%}

%% Finally, 3 digit testing
%{
paramStruct.labelTypes = [2, 7, 9];
paramStruct.numLabelTypes = 3;
paramStruct.normalizeYs = 0;
load('mnist_test.mat')
[Xts, yts] = parseData(data, labels, 'Test-2-7-9.mat', paramStruct);

load('Test-2-7-9.mat');
Xts = input_data.X;
yts = input_data.y;

% arrange the weights
load('1000, 2-7-[3-1].mat')
weights27 = results.weights;
load('1000, 7-9-[3-1].mat')
weights79 = results.weights;
load('1000, 2-9-[3-1].mat')
weights29 = results.weights;

testArrangements = {[2,7], [7,9], [2,9]};
guesses = zeros(numel(yts), 1);

for i = 1:numel(yts)
    % Find the highest confidence
    conf = 0;
    w = 1;
    arr = 1;
    temparr = 1;
    temp = 0;
    [ypred1, yall] = testMLP(Xts(:,i), weights27, paramStruct);
    if ypred1 > 0.5
        conf = 1 - ypred1;
        arr = 2;
    else
        conf = ypred1;
    end
    
    [ypred2, yall] = testMLP(Xts(:,i), weights29, paramStruct);
    if ypred2 > 0.5
        temp = 1 - ypred2;
        temparr = 2;
    else
        temp = ypred2;
        temparr = 1;
    end
    
    if temp < conf
        w = 2;
        conf = temp;
        arr = temparr;
    end
    
    [ypred3, yall] = testMLP(Xts(:,i), weights29, paramStruct);
    if ypred3 > 0.5
        temp = 1 - ypred3;
        temparr = 2;
    else
        temp = ypred3;
        temparr = 1;
    end
    
    if temp < conf
        w = 3;
        conf = temp;
        arr = temparr;
    end
    
    if w == 1
        ypred = ypred1;
    elseif w == 2
        ypred = ypred2;
    else
        ypred = ypred3;
    end
    

    guesses(i) = testArrangements{w}(arr);
end

disp('Classifying 3 digits error:')
disp(sum(guesses ~= yts) / numel(yts));
%}
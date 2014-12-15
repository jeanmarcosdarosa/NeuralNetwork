function [ypred, yall] = testMLP( X, weights, paramStruct )
    % Xtr: dataDim x n feature matrix 
    %       where dataDim is the #features and n is #data pts
    %       Xtr should be standardized, 
    %       i.e. mean(Xtr(d,:)) = 0,
    %            std(Xtr(d,:))  = 1
    % ytr: n x 1 label vector. ytr(i) in {0, 1}. 
    % param: structure of parameters
    %       .layersDims: 1 x L size array, 
    %           L is the number of hidden+output layers
    %           siz(l) is the size of hidden layer l. 
    %       .lambda: weight decay
    %       .learningRate: learning rate for updating the weights
    %       .maxIterations: maximum number of iterations
    %
    % ypred: n x 1 vector of prediction. ypred(i) in {0, 1}. 
    % yall:  cell array of hidden activations. 
    %       yall{l} is n x 1 vector of activation at layer l. 
    %       yall{L} is the same as ypred. 
    
    % assert the number of layers and the number of weight vectors match
    L = length( paramStruct.layersDims );
    assert( L == length(weights) );

    yall = cell( L, 1 );
    % number of data points
    n = size(X, 2);
    % input to the previous layer (set to X for layer 0)
    yprev = [ones(1, n); X];
    for l=1:L
       yall{l} = logistic( weights{l} * yprev );
       % append ones to the features (keeps biases constant at one)
       yprev = [ones(1, n); yall{l}];
    end
    ypred = yall{L};
end

function output = logistic( input )  % sigmoid
    output = 1 ./ ( 1 + exp( -input ) );   
end
function trainThenTest(Xtr, ytr, Xts, yts, filename, paramStruct)
    % Xtr: dataDim x n feature matrix 
    %       where dataDim is the #features and n is #data pts
    %       Xtr should be standardized, 
    %       i.e. mean(Xtr(d,:)) = 0,
    %            std(Xtr(d,:))  = 1
    % ytr: n x 1 label vector. ytr(i) in {0, 1}. 
    % Xts: Test set feature matrix
    % yts: Test set labels
    % filename: File to save results to
    % param: structure of parameters
    %       .layersDims: 1 x L size array, 
    %           L is the number of hidden+output layers
    %           siz(l) is the size of hidden layer l. 
    %       .lambda: weight decay
    %       .learningRate: learning rate for updating the weights
    %       .maxIterations: maximum number of iterations
    %
    % weights: L x 1 cell array
    %       weights{l} is a p_l x [1+p_{l-1}] weight matrix that transforms
    %       input at layer l-1 to layer l. l = 1,2,..L. 
    
    % First we train on the training set to get weights.
    weights = trainMLP(Xtr, ytr, filename, paramStruct);
    
    % How did we do? Test on the testing set.
    load(filename, 'results');
    ypred = testMLP(Xts, weights, paramStruct)' > 0.5;
    fprintf('Testing accuracy = %.2f%%\n', 100*mean((yts == ypred)));
    
    % Save our results
    results.testPred = ypred;
    results.testAccuracy = 100*mean((yts == ypred));
    results.Xtr = Xtr;
    results.ytr = ytr;
    results.Xts = Xts;
    results.yts = yts;
    results.layers = paramStruct.layersDims;
    results.learningRate = paramStruct.learningRate;
    results.nLabelTypes = paramStruct.nLabelTypes;
    results.labelTypes = paramStruct.labelTypes;
    save(filename, 'results');
 
end



function weights = trainMLP(Xtr, Ytr, filename, paramStruct)
    % Xtr: dataDim x n feature matrix 
    %       where dataDim is the #features and n is #data pts
    %       Xtr should be standardized, 
    %       i.e. mean(Xtr(d,:)) = 0,
    %            std(Xtr(d,:))  = 1
    % ytr: n x 1 label vector. ytr(i) in {0, 1}. 
    % filename: File to save weights to
    % param: structure of parameters
    %       .layersDims: 1 x L size array, 
    %           L is the number of hidden+output layers
    %           siz(l) is the size of hidden layer l. 
    %       .lambda: weight decay
    %       .learningRate: learning rate for updating the weights
    %       .maxIterations: maximum number of iterations
    %
    % weights: L x 1 cell array
    %       weights{l} is a p_l x [1+p_{l-1}] weight matrix that transforms
    %       input at layer l-1 to layer l. l = 1,2,..L. 
    
    %% SET PARAMETERS

    % number of layers of weights
    L = length( paramStruct.layersDims ); 
    % dimension of the input data 
    [dataDim, num_points] = size( Xtr );
    
    % Training errors
    %{
    if paramStruct.trackErrors == 1
        errors = zeros(1, paramStruct.maxIterations + 1);
    end
    %}
    
    % Container for the signal values going forward through the Network
    X_signal = cell(1, L + 1);
    % Container for the signal values going forward through the Network
    S_signal = cell(1, L);
    % Container for the delta values going backwards through the Network
    delta = cell(1, L);
    % Container for the gradient updates
    wgrad = cell(1, L);
    
    %{
    % Container for weights to keep track of L1, L2.
    if paramStruct.trackWeights == 1
        paramStruct.trainingWeightList = cell(1, paramStruct.maxIterations);
    end
    %}
    
    [p, n] = size(Xtr);
    siz = [ p, paramStruct.layersDims ];
    
    % Initialize matrices to all zero in order to avoid re-allocation
    X_signal{1} = zeros( 1 + dataDim, num_points );
    for l = 1:L-1
        S_signal{l + 1} = zeros( paramStruct.layersDims(l), num_points );
        X_signal{l + 1} = zeros( 1 + paramStruct.layersDims(l), num_points );
    end
    S_signal{L + 1} = zeros( paramStruct.layersDims(L), num_points );
    X_signal{L + 1} = zeros( paramStruct.layersDims(L), num_points );

    for l = 1:L
        delta{l} = zeros( paramStruct.layersDims(l), num_points );
    end

    wgrad{1} = zeros( paramStruct.layersDims(1), 1 + dataDim );
    for l = 1:L-1
        wgrad{l + 1} = zeros( paramStruct.layersDims( l + 1 ), ...
            1 + paramStruct.layersDims(l) );
    end

    signalsStruct.X_signal = X_signal;
    signalsStruct.S_signal = S_signal; 
    signalsStruct.delta = delta;
    signalsStruct.wgrad = wgrad;

    % Size of each layer (including input and output ) augmented with bias
    % neurons. First layer has number of neurons equal to the dimension of
    % input data + 1, rest of layers are determined by input parameter 
    layersDims = [ dataDim + 1, ...
                   paramStruct.layersDims(1:end-1) + 1, ...
                   paramStruct.layersDims(end) ];
    % Variable for counting time
    tStart = tic;

    %% WEIGHT INITIALIZATION

    % Each weight matrix has number of rows given by the number of
    % neurons in the next layer (exluding the bias unit) and number of
    % columns corresponding to the number of neurons in the previous layer.
    % This way the signal at the next layer can be computed as a matrix
    % product x(l) = W(l)*x(l-1).
    weights = cell(1, L);

    initWeightsMagnitude = 0.05;

    % Initialization of all layers except output
    for l = 1:L-1

        % Random initialization for each weight of the network between 
        % -0.05 and +0.05 has been implemented.
        weights{l} = -initWeightsMagnitude + ...
           2*initWeightsMagnitude*rand( layersDims(l + 1) - 1, layersDims(l) );

        % The bias weights are initialized to the maximum value of the
        % weights in the hidden units.
        % References: 
        %
        %  - Palubinskas, G. Data-driven Weight Initialization of 
        %    Back-propagation for Pattern Recognition. Pro. of the Int. Conf. 
        %    on Artificial Neural Networks, vol. 2. pp. 851-854, 1994.
        %
        %  - Fernández-Redondo, Mercedes, and Carlos Hernández-Espinosa."Weight
        %    initialization methods for multilayer feedforward." ESANN. 2001.
        [value, ~] = max(weights{l}(:));
        weights{l}(:,1) = value; 

    end
    % Initialization of output layer
    weights{L} = -initWeightsMagnitude + ...
            2*initWeightsMagnitude*rand( layersDims(L + 1), layersDims(L) );

    [value, ~] = max(weights{L}(:));
    weights{L}(:,1) = value; 

    %% RUN MAX_ITERATIONS OF BACK PROPAGATION ALGORITHM
    %{
    % Get predictions from network by thresholding at 0.5 to decide between
    % class Y = 1 and Y = 0
    ypred = testMLP( Xtr, weights, paramStruct )' > 0.5;
    if paramStruct.trackErrors == 1
        errors(1) = MSE(Ytr, ypred, num_points);
    end
    fprintf('Initial training accuracy = %.2f%%\n', 100 * mean((Ytr == ypred)));
    %}
    % implement gradient descent
    maxIterations = paramStruct.maxIterations;
    for iter = 1:maxIterations
        % Weight tracking
        if paramStruct.trackWeights == 1
            paramStruct.trainingWeightList{iter} = weights;
        end
        perm_idx = randperm( num_points );
        for ex = 1:num_points

            x_tr = Xtr(:,perm_idx(ex));
            y_tr = Ytr(perm_idx(ex));

            % compute the value for weight update (dL/dW for all weights)
            wgrad = backprop( x_tr, y_tr, weights, paramStruct, signalsStruct );
                        
            % update the weights at all layers
            for l = 1:L
                %{
                % L1 regularization
                if paramStruct.L1 == 1 
                    wgrad{l} = wgrad{l} + [zeros(siz(l+1), 1) paramStruct.gamma*L1( weights{l}(:,2:end), paramStruct)];
                end

                % L2 regularization
                if paramStruct.L2 == 1
                    wgrad{l} = wgrad{l} + [zeros(siz(l+1), 1) paramStruct.gamma*L2(weights{l}(:,2:end), paramStruct)];
                end
                %}
                weights{l} = weights{l} - ...
                    paramStruct.learningRate * wgrad{l};
            end
        end

        % every n iterations or so, print some intermediate results
        % e.g iteration number, training error, weights, etc.
        if mod(iter,500) == 0

            S = [ 'Completed ', num2str(iter), ' iterations in ', ...
                num2str(toc(tStart)), ' seconds.'];
            disp(S)
            ypred = testMLP( Xtr, weights, paramStruct )' > 0.5;
            fprintf('Current training accuracy = %.2f%%\n', ...
                100*mean((Ytr == ypred)));
        end
        %{
        if paramStruct.trackErrors == 1
            errors(iter + 1) = MSE(Ytr, ypred, num_points);
        end
        %}
    end
    
    % save our results
    %results.trainingWeightList = paramStruct.trainingWeightList;
    results.L1 = paramStruct.L1;
    results.L2 = paramStruct.L2;
    results.trainRunTime = toc(tStart);
    %results.trainingErrors = errors;
    results.weights = weights;
    results.trainAccuracy = 100*mean((Ytr == ypred));
    results.trainIterations = paramStruct.maxIterations;
    save(filename, 'results');       
end

function wgrad = backprop( x_tr, y_tr, weights, paramStruct, signalsStruct )
    % every iteration consist of a forward pass to compute the loss and a
    % backwards pass to compute all the deltas for the weights update

    X_signal = signalsStruct.X_signal;
    S_signal = signalsStruct.S_signal;
    delta = signalsStruct.delta;
    wgrad = signalsStruct.wgrad;
    
    % number of layers of weights
    L = length( paramStruct.layersDims );
    % dimension of the input data 
    n = size( x_tr, 2 );

    %% FORWARD STEP
    % Compute all signal values
    % Augment Xtr for bias neuron
    X_signal{1} = [ones(1,n); x_tr];
    for l = 1:L-1
        S_signal{l + 1} = weights{l} * X_signal{l};
        X_signal{l + 1} = [ ones(1,n); 1 ./ ( 1 + exp( -S_signal{l + 1} ) )];
    end
    S_signal{L + 1} = weights{L} * X_signal{L};
    X_signal{L + 1} = 1 ./ ( 1 + exp( - S_signal{L + 1} ) );

    %% BACKWARD STEP
    % Compute all delta values

    % Using square error as cost function
    % delta for output layer
    delta{L} = ...
        2 * ( X_signal{L + 1} - y_tr' ) .* ...
        ( X_signal{L + 1} ) .* ( 1 - X_signal{L + 1} );

    % delta for intermediate layers
    for l = 1:L-1
        delta{L-l} = ( X_signal{L - l + 1}(2:end) ) .* ... 
            ( 1 - X_signal{L - l + 1}(2:end) ) .*...
            ((weights{L-l+1}(:,2:end))' * delta{L-l+1});
    end

    %% COMBINATION STEP
    for l = 1:L
        wgrad{l} =  delta{l} * X_signal{l}';
    end
    
end

function w = L1(w, paramStruct)
    for i = 1:numel(w)
        w(i) = w(i) - paramStruct.gamma * sign(w(i));
    end
end

function w = L2(w, paramStruct)
    for i = 1:numel(w)
        w(i) = w(i) - w(i) * paramStruct.gamma;
    end
end



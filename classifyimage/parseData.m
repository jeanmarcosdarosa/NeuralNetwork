function [X, y] = parseData(data, labels, filename, paramStruct)
    % data: Input image data, each image is an NxN matrix and are arranged
    %       in an array.
    % labels: Labels of each image for the training set.
    % filename: Name of file to save the data to
    % param: The structure of parameters.
    %        .labelTypes: The types of labels we are interested in
    %                     classifying. NOTE: This can be a fraction of the
    %                     provided data pairs.
    %        .numLabelTypes: How many types of labels we are classifying.
                          
    % Initialize and allocate our data matrices
    N = 0; % Total number of data points we care about
    for i = 1:paramStruct.numLabelTypes
        N = N + sum(labels == paramStruct.labelTypes(i));
    end
    
    X = zeros(numel(data(:,:,1)), N); % datadim by N
    y = zeros(N, 1);
   
    % Parse the data by category, for example, if we are classifying 0s and
    % 1s, then our ytrList will be only 0s and 1s, and their corresponding
    % images will be placed into XtrList. 
    count = 1;
    for pt = 1:length(data)
        for i = 1:paramStruct.numLabelTypes 
            if paramStruct.labelTypes(i) == labels(pt)
                % Standardize inputs
                X(:,count) = zscore(reshape(data(:,:,pt), [], 1));
                if paramStruct.normalizeYs == 1
                    y(count) = paramStruct.nLabelTypes(i);
                else
                    y(count) = paramStruct.labelTypes(i);
                end
                count = count + 1;
            end
        end
    end
 
    % Save to a file - useful for debugging
    input_data.X = X;
    input_data.y = y;
    save(filename, 'input_data');
end
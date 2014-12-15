function mse = MSE(expected, results, numPts)
    % This function calculates the mean squared error of a result
    mse = 0;
    for i = 1:numPts
        mse = mse + (expected(i) - results(i))^2;
    end
    mse = mse / numPts;
end
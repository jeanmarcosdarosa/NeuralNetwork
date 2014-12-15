function [images, errors] = reconstructImage(U, S, V, image, A, param)
    images = cell(1, numel(param.R));
    [imageX, imageY] = size(image);
    sizeX = param.sampleSize(1);
    sizeY = param.sampleSize(2);
    A_approxes = cell(1, numel(param.R));
    errors = cell(1, numel(param.R));
    
    % Rearrange each col to an 8 by 8 patch and then overlay them...
    averages = zeros(imageX, imageY);

    for rIndex = 1:numel(param.R)
        count = 1;
        images{rIndex} = zeros(imageX, imageY);
        R = param.R(rIndex);
        A_approxes{rIndex} = U(:,1:R) * S(1:R,1:R) * transpose(V(:,1:R));

        % Add overlapping sampled images together
        for row = (sizeX/2 + 1):(imageX - sizeX/2 + 1)
            for col = (sizeY/2 + 1):(imageY - sizeY/2 + 1)
                patch = reshape(A_approxes{rIndex}(:,count), [sizeX, sizeY]);
                images{rIndex}((row-sizeX/2):(row+sizeX/2-1), ...
                    (col-sizeY/2):(col+sizeY/2-1)) = ...
                    images{rIndex}((row-sizeX/2):(row+sizeX/2-1), ...
                    (col-sizeY/2):(col+sizeY/2-1)) + patch; 

                if rIndex == 1
                    averages((row-sizeX/2):(row+sizeX/2-1), ...
                        (col-sizeY/2):(col+sizeY/2-1)) = ...
                        averages((row-sizeX/2):(row+sizeX/2-1), ...
                        (col-sizeY/2):(col+sizeY/2-1)) + 1;
                end

                count = count + 1;
            end
        end

        % Lastly, average the portions that have overlap
        images{rIndex} = images{rIndex} ./ averages;
    end
end
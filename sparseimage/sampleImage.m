function s = sampleImage(image, param) 
    sizeX = param.sampleSize(1);
    sizeY = param.sampleSize(2);
    [imageX, imageY] = size(image);
    numSamples = (imageX - 0.5*sizeX) * (imageY - 0.5*sizeY);
    s = zeros(sizeX * sizeY, numSamples);

    count = 1;
    for row = (sizeX/2 + 1):(imageX - sizeX/2 + 1)
        for col = (sizeY/2 + 1):(imageY - sizeY/2 + 1)
            s(:,count) = im2col(image((row-sizeX/2):(row+sizeX/2-1), ...
                (col-sizeY/2):(col+sizeY/2-1)), [sizeX sizeY]);
            count = count + 1;
        end
    end
    
    results.samples = s;
end
-- Eduardo Romera,
-- May 2017.
-- This code is called from train/test to get batches and perform data augmentations
----------------------------------------------------------------------

require 'image'

local cityscapesMean = torch.FloatTensor{73.633/255.0, 83.367/255.0, 72.867/255.0}	--hardcoded value although it is calculated at dataset loading

function getBatchTrain (t, trainData, shuffle)
    local x,yt = getMiniBatch (t, trainData, shuffle)
    return preprocessTrain (x, yt)	--perform data augmentation and so on
end

function getBatchTest (t, testData, shuffle)	--shuffle is just dummy ordered list in test
    local x,yt = getMiniBatch (t, testData, shuffle)

    return preprocessTest(x,yt)	--subtract mean if needed, etc
end

function getMiniBatch (t, dataset, shuffle)	--dataset is either trainData or testData
    -- create mini batch
    local x = torch.Tensor(opt.batchSize, opt.channels, opt.imHeight, opt.imWidth)
    local yt = torch.Tensor(opt.batchSize, opt.labelHeight, opt.labelWidth)
    local idx = 1

    if (opt.loadMode==0) then
        for i = t,t+opt.batchSize-1 do
            x[idx] = dataset.data[shuffle[i]]:float()	/ 255.0	
            yt[idx] = dataset.labels[shuffle[i]]:float() 
            idx = idx + 1
        end
    else
        for i = t,t+opt.batchSize-1 do
            local imgPath = dataset.data[shuffle[i]]
            local gtPath = dataset.labels[shuffle[i]]

            local dataTemp = image.load(imgPath)
            local img = image.scale(dataTemp,opt.imWidth, opt.imHeight)

            -- label image data are resized to be [1,nClasses] in [0 255] scale:
            local labelIn = image.load(gtPath, 1, 'byte')
            local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

            if (opt.decoder ~= true) then
                labelFile = image.scale(labelFile, opt.labelWidth, opt.labelHeight, 'simple')
            end

            labelFile:apply(function(x) return classMap[x][1] end)

            x[idx] = img
            yt[idx] = labelFile
            idx = idx + 1
        end
    end

    return x, yt
end
		
function preprocessTrain (x, yt)

    if (opt.dataset == 'cs' and opt.subtractmean) then
        for i=1,3 do x[{{},i}]:csub(cityscapesMean[i]) end
    end

    if (opt.augment) then
        x, yt = augment(x,yt)
    end

    return x, yt

end

function preprocessTest (x, yt)

    if (opt.subtractmean) then
        for i=1,3 do x[{{},i}]:csub(cityscapesMean[i]) end
    end

    return x,yt
end


function augment (x, yt)
    for idx = 1, opt.batchSize do
        local x_aux = x[idx]--:clone()   --clone not necessary cos translate/hflip generate new tensor
        local yt_aux = yt[idx]--:clone()

        --random translations + hflip
        local transX = torch.random(-2, 2)  --possible: -2, -1, 0, 1, 2
        local transY = torch.random(-2, 2)
        local hflip = torch.random(0,1)

        --translation - random translates with zero-padding between -2 and +2 
        x_aux = image.translate(x_aux, transX, transY)
        if (opt.decoder) then  --if enc, translation not applied to yt cos encoder output is 1/8 size

            yt_aux = image.translate(yt_aux, transX, transY)

            --fix zero-padding to 1-padding only for labels (as ignore label is set to 1)
            if ((transX == 1) or (transX == 2)) then
                yt_aux[{{},{1,transX}}] = 1
            elseif ((transX == -1) or (transX == -2)) then
                yt_aux[{{},{yt_aux:size(2)+transX+1,yt_aux:size(2)}}] = 1
            end
            if ((transY == 1) or (transY == 2)) then
                yt_aux[{{1,transY},{}}] = 1
            elseif ((transY == -1) or (transY == -2)) then
                yt_aux[{{yt_aux:size(1)+transY+1,yt_aux:size(1)},{}}] = 1
            end
        end

        if (hflip==1) then
        x_aux = image.hflip(x_aux)
        yt_aux = image.hflip(yt_aux)
        end

        x[idx] = x_aux
        yt[idx] = yt_aux
    end

    return x, yt
end



----------------------------------------------------------------------
-- Eduardo Romera,
-- December 2016
-- Cityscape data loader, load file lists (train.txt and val.txt) and calculate mean/std and histClasses. Called in loadMode=1
----------------------------------------------------------------------

require 'torch'   
require 'image'   

paths.dofile('cityscapeClassMaps.lua')	--load the class indices


function loadDataset()

    local trainFile = 'data/train.txt'
    local testFile = 'data/val.txt'

    function getPath(filepath)
        print("Extracting file names from: " .. filepath)
        local file = io.open(filepath, 'r')
        local imgPaths = {}
        local gtPaths = {}
        local fline = file:read()
        while fline ~= nil do
            local col1 = opt.datapath .. '/leftImg8bit/' .. fline .. '_leftImg8bit.png'
            local col2 = opt.datapath .. '/gtFine/' .. fline .. '_gtFine_labelIds.png'
            table.insert(imgPaths, col1)
            table.insert(gtPaths, col2)
            fline = file:read()
        end
        return imgPaths, gtPaths
    end


    local imgPaths, gtPaths = getPath(trainFile)
    local trsize = #imgPaths

    local   trainData = { 
        data = imgPaths,
        labels = gtPaths,
        size = function() return trsize end
    }

	imgPaths, gtPaths = getPath(testFile)
	local tesize = #imgPaths

    local	   testData = { 
        data = imgPaths,
        labels = gtPaths,
        size = function() return tesize end
    }

    -- calculate histClasses and mean/std of the dataset, or load from previous calculations
    local cachePath = opt.cachepath	.. 'cityscapes/'--modificado para poder guardarlo como yo quiera con el path
    paths.mkdir(cachePath)	
    local imSizeString = opt.imWidth .. 'x' .. opt.imHeight
    local pathCacheData = cachePath .. 'cachedata_fine_imgs_' .. imSizeString .. '.t7'
    local trainMean, trainStd, histClasses, histClassesEncoder

    if paths.filep(pathCacheData) then
        print '==> Loading dataset mean/std and class balances (cache data)'
        local cacheData = torch.load(pathCacheData)
        trainMean = cacheData.trainMean
        trainStd = cacheData.trainStd
        histClasses = cacheData.histClasses
        histClassesEncoder = cacheData.histClassesEncoder
    else

        trainMean = torch.FloatTensor(3)
        trainStd = torch.FloatTensor(3)
        local meanEstimate = {0,0,0}
        local stdEstimate = {0,0,0}
        histClasses = torch.FloatTensor(#classes):zero()
        histClassesEncoder = torch.FloatTensor(#classes):zero()

        print '==> Calculating dataset mean/std and class balances (cache data)'
        for i = 1,trsize do
            local imgPath = trainData.data[i]
            local gtPath = trainData.labels[i]

            local dataTemp = image.load(imgPath)--gm.load(imgPath, 'byte')
            local img = image.scale(dataTemp,opt.imWidth, opt.imHeight)

            for j=1,3 do
                meanEstimate[j] = meanEstimate[j] + img[j]:mean()
                stdEstimate[j] = stdEstimate[j] + img[j]:std()
            end

            -- label image data are resized to be [1,nClasses] in [0 255] scale:
            local labelIn = image.load(gtPath, 1, 'byte') --gm.load(gtPath, 'byte')[1]
            local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()

            labelFile:apply(function(x) return classMap[x][1] end)

            local labelFileEncoder = image.scale(labelFile, opt.labelWidth, opt.labelHeight, 'simple'):float()

            histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)
            histClassesEncoder = histClassesEncoder + torch.histc(labelFileEncoder, #classes, 1, #classes)

            xlua.progress(i, trsize)
        end

        for j=1,3 do
            trainMean[j] = meanEstimate[j] / trsize
            trainStd[j] = stdEstimate[j] / trsize
        end

        print('\t\t Saving dataset cache data')   
        local cacheData = {
            histClasses = histClasses,
            histClassesEncoder = histClassesEncoder,
            trainMean = trainMean,
            trainStd = trainStd
        }
        torch.save(pathCacheData, cacheData)

        if (paths.filep(pathCacheData) == false) then print("ERROR: Cache directory is not accessible") end
    end

    -- Exports
    opt.dataClasses = classes
    opt.dataconClasses  = conClasses

    if (opt.decoder) then
        opt.datahistClasses = histClasses
    else
        opt.datahistClasses = histClassesEncoder
    end

    print ('printing train set stats:')
    print(string.format('train data, mean: [%.3f, %.3f, %.3f]', trainMean[1], trainMean[2], trainMean[3]))
    print(string.format('train data, standard deviation: [%.3f, %.3f, %.3f]', trainStd[1], trainStd[2], trainStd[3]))

    return {
        trainData = trainData,
        testData = testData,
        mean = trainMean,
        std = trainStd
    }
end




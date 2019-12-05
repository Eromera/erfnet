----------------------------------------------------------------------
-- Cityscape data loader,
-- Originally written by Abhishek Chaurasia,
--
-- Modified by Eduardo Romera:
-- load images in 8bit mode (saves memory) and save cache in separate files for encoder/decoder. Called in loadMode=0
-- May 2017.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

paths.dofile('cityscapeClassMaps.lua')	--load the class indices

function loadDataset()
    --Force ByteTensor to load images in Byte mode instead of Float (1/4 size in memory)
    torch.setdefaulttensortype('torch.ByteTensor')

    local trsize, tesize
    trsize = 2975 -- cityscape train images
    tesize = 500  -- cityscape validation images

    -- saving training histogram of classes
    local histClasses = torch.FloatTensor(#classes):zero()
    local histClassesEncoder = torch.FloatTensor(#classes):zero()

    --------------------------------------------------------------------------------
    print '==> loading cityscapes dataset'
    local trainData, testData
    local loadedFromCache = false
    local cachePath = opt.cachepath	.. 'cityscapes/'
    paths.mkdir(cachePath)

    local sizeString = opt.imWidth .. 'x' .. opt.imHeight
    local sizeStringEncoder = (opt.imWidth * 0.125) .. 'x' .. (opt.imHeight * 0.125)

    local pathCacheImgs = cachePath .. 'fine_imgs_' .. sizeString .. '.t7'
    local pathCacheLabels = cachePath .. 'fine_labels_' .. sizeString .. '.t7'
    local pathCacheLabelsEncoder = cachePath .. 'fine_labels_' .. sizeStringEncoder .. '.t7'

    if paths.dirp(cachePath) and paths.filep(pathCacheImgs) and paths.filep(pathCacheLabels) and paths.filep(pathCacheLabelsEncoder) then

        print ('\t loading ' .. pathCacheImgs)
        local dataCacheImgs = torch.load(pathCacheImgs)
        local dataCacheLabels 
        if (opt.decoder) then
            print ('\t loading ' .. pathCacheLabels)
            dataCacheLabels = torch.load(pathCacheLabels)
        else
            print ('\t loading ' .. pathCacheLabelsEncoder)
            dataCacheLabels = torch.load(pathCacheLabelsEncoder)      
        end

        trainData = { 
            data = dataCacheImgs.trainImgs,
            labels = dataCacheLabels.trainLabels,
            size = function() return trsize end
        }

        testData = { 
            data = dataCacheImgs.testImgs,
            labels = dataCacheLabels.testLabels,
            size = function() return tesize end
        }

        histClasses = dataCacheLabels.histClasses
        trainMean = dataCacheImgs.trainMean
        trainStd = dataCacheImgs.trainStd
        testMean = dataCacheImgs.testMean
        testStd = dataCacheImgs.testStd

        loadedFromCache = true
        dataCache = nil
        collectgarbage()
    else
        local function has_image_extensions(filename)
        local ext = string.lower(path.extension(filename))

        -- compare with list of image extensions
        local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
        for i = 1, #img_extensions do
            if ext == img_extensions[i] then
                return true
            end
        end
            return false
        end



        -- initialize data structures:
        trainData = {
            data = torch.ByteTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
            labels = torch.ByteTensor(trsize, opt.imHeight, opt.imWidth),
            labelsEncoder = torch.ByteTensor(trsize, opt.imHeight*0.125, opt.imWidth*0.125),
            size = function() return trsize end
        }

        testData = {
            data = torch.ByteTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
            labels = torch.ByteTensor(tesize, opt.imHeight, opt.imWidth),
            labelsEncoder = torch.ByteTensor(tesize, opt.imHeight*0.125, opt.imWidth*0.125),
            size = function() return tesize end
        }

        print('==> loading training files');

        local dpathRoot = opt.datapath .. '/leftImg8bit/train/'

        assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. opt.datapath)
        --load training images and labels:
        local c = 1
        for dir in paths.iterdirs(dpathRoot) do
            local dpath = dpathRoot .. dir .. '/'
            for file in paths.iterfiles(dpath) do

                -- process each image
                if has_image_extensions(file) and c <= trsize then
                    local imgPath = path.join(dpath, file)

                    --load training images:
                    local dataTemp = image.load(imgPath, 3, 'byte')
                    trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)

                    -- Load training labels:
                    -- Load labels with same filename as input image.
                    imgPath = string.gsub(imgPath, "leftImg8bit", "gtFine")
                    imgPath = string.gsub(imgPath, ".png", "_labelIds.png")

                    -- label image data are resized to be [1,nClasses] in [0 255] scale:
                    local labelIn = image.load(imgPath, 1, 'byte')
                    local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()
                    labelFile:apply(function(x) return classMap[x][1] end)

                    local labelFileEncoder = image.scale(labelFile, opt.imWidth*0.125, opt.imHeight*0.125, 'simple'):float()

                    -- Syntax: histc(data, bins, min, max)
                    histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)
                    histClassesEncoder = histClassesEncoder + torch.histc(labelFileEncoder, #classes, 1, #classes)

                    -- convert to int and write to data structure:
                    trainData.labels[c] = labelFile:byte()
                    trainData.labelsEncoder[c] = labelFileEncoder:byte()

                    c = c + 1
                    if c % 20 == 0 then
                        xlua.progress(c, trsize)
                    end
                    collectgarbage()
                end
            end
        end
        print('')

        print('==> loading testing files');
        dpathRoot = opt.datapath .. '/leftImg8bit/val/'

        assert(paths.dirp(dpathRoot), 'No testing folder found at: ' .. opt.datapath)
        -- load test images and labels:
        local c = 1
        for dir in paths.iterdirs(dpathRoot) do
            local dpath = dpathRoot .. dir .. '/'
            for file in paths.iterfiles(dpath) do

                -- process each image
                if has_image_extensions(file) and c <= tesize then
                    local imgPath = path.join(dpath, file)

                    --load training images:
                    local dataTemp = image.load(imgPath, 3, 'byte')
                    testData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)

                    -- Load validation labels:
                    -- Load labels with same filename as input image.
                    imgPath = string.gsub(imgPath, "leftImg8bit", "gtFine")
                    imgPath = string.gsub(imgPath, ".png", "_labelIds.png")

                    -- load test labels:
                    -- label image data are resized to be [1,nClasses] in in [0 255] scale:
                    local labelIn = image.load(imgPath, 1, 'byte')
                    local labelFile = image.scale(labelIn, opt.imWidth, opt.imHeight, 'simple'):float()
                    labelFile:apply(function(x) return classMap[x][1] end)

                    local labelFileEncoder = image.scale(labelFile, opt.imWidth*0.125, opt.imHeight*0.125, 'simple')

                    -- convert to int and write to data structure:
                    testData.labels[c] = labelFile:byte()
                    testData.labelsEncoder[c] = labelFileEncoder:byte()

                    c = c + 1
                    if c % 20 == 0 then
                        xlua.progress(c, tesize)
                    end
                    collectgarbage()
                end
            end
        end
    end

    ----------------------------------------------------------------------
    print '==> verify statistics'

    -- It's always good practice to verify that data is properly normalized.
    --[[
    if (trainMean == nil) then
        trainMean = torch.FloatTensor(3)
        trainStd = torch.FloatTensor(3)
        testMean = torch.FloatTensor(3)
        testStd = torch.FloatTensor(3)

        for i = 1, opt.channels do
            local trainMeani = torch.mean(trainData.data[{ {},i }]:float())
            local trainStdi = torch.std(trainData.data[{ {},i }]:float())

            local testMeani = torch.mean(testData.data[{ {},i }]:float())
            local testStdi = torch.std(testData.data[{ {},i }]:float())

            print('training data, channel-'.. i ..', mean: ' .. trainMeani)
            print('training data, channel-'.. i ..', standard deviation: ' .. trainStdi)

            print('test data, channel-'.. i ..', mean: ' .. testMeani)
            print('test data, channel-'.. i ..', standard deviation: ' .. testStdi)

            trainMean[i] = trainMeani
            trainStd[i] = trainStdi
            testMean[i] = testMeani
            testStd[i] = testStdi
        end

        if (loadedFromCache) then  --modify cache file to resave this
            print ('Rewriting cachefile to save stats')
            loadedFromCache = false --to force rewrite
        end
        else
            print ('printing stats from file:')
            print(string.format('train data, mean: [%.3f, %.3f, %.3f]', trainMean[1], trainMean[2], trainMean[3]))
            print(string.format('train data, standard deviation: [%.3f, %.3f, %.3f]', trainStd[1], trainStd[2], trainStd[3]))

            print(string.format('test data, mean: [%.3f, %.3f, %.3f]', testMean[1], testMean[2], testMean[3]))
            print(string.format('test data, standard deviation: [%.3f, %.3f, %.3f]', testStd[1], testStd[2], testStd[3]))

        end
    --]]
    -----------------------------------------------------------
    --save cache file in case it wasnt saved

    if paths.dirp(opt.cachepath) and not loadedFromCache then
        print('==> saving data to cache: ' .. cachePath)

        print('\t\t saving imgs')   
        local dataCacheImgs = {

            trainImgs = trainData.data,
            testImgs = testData.data,

            trainMean = trainMean,
            trainStd = trainStd,
            testMean = testMean,
            testStd = testStd
        }
        torch.save(pathCacheImgs, dataCacheImgs)
        dataCacheImgs = nil

        print('\t\t saving decoder labels')   
        local dataCacheLabels = {
            trainLabels = trainData.labels,
            testLabels = testData.labels,
            histClasses = histClasses
        }
        torch.save(pathCacheLabels, dataCacheLabels)
        dataCacheLabels = nil

        print('\t\t saving encoder labels')   
        local dataCacheLabelsEncoder = {
            trainLabels = trainData.labelsEncoder,
            testLabels = testData.labelsEncoder,
            histClasses = histClassesEncoder
        }
        torch.save(pathCacheLabelsEncoder, dataCacheLabelsEncoder)

        dataCacheLabels = nil
        dataCacheLabelsEncoder = nil

        --Once saved: make it load only encoder or decoder labels:
        if (opt.decoder ~= true) then
            trainData.labels = trainData.labelsEncoder
            testData.labels = testData.labelsEncoder
            histClasses = histClassesEncoder
        end
        trainData.labelsEncoder = nil
        testData.labelsEncoder = nil
        histClassesEncoder = nil

        collectgarbage()
    end

	---------------------------------------------------------------------

    -- Exports
    opt.dataClasses = classes
    opt.dataconClasses  = conClasses
    opt.datahistClasses = histClasses

    --Set this back to Float tensor for the rest of the code
    torch.setdefaulttensortype('torch.FloatTensor')

    return {
        trainData = trainData,
        testData = testData,
        mean = trainMean,
        std = trainStd
    }
end

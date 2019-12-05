-- Eduardo Romera,
-- May 2017.
-- Main code for training a model
----------------------------------------------------------------------

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cudnn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)
cudnn.benchmark = true  

epoch = 1 --global variable

-------------------------------------------------------

opt = paths.dofile('loadOptions.lua')	--parse and load arguments

-------------------------------------------------------

paths.dofile('utils.lua')	--define some general functions (pause, saveModel...)

-------------------------------------------------------
--Load Dataset:
if (opt.loadMode == 1) then
    --in loadmode==1 we preload the train.txt and val.txt list of paths and load each file at training time
    require 'data/loadCityscapeList'	--only cityscapes mode has been developed so far
else
    --in loadmode==0 we preload all data in memory (ram must fit entire dataset)
    require 'data/loadCityscape8bit'
end

data = loadDataset()
assert(data, "Dataset wasnt loaded correctly (data==nil)")



-------------------------------------------------------

--globals to control best iterations
trainError = 1e10	--current
prevTrainError = 1e10   --for lrDecayAuto
bestTrainError = 1e10
bestunion = 0 --variable for saving best IoU model
bestTestError = 1e10 --variable for saving best IoU model

optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
}

-------------------------------------------------------

t = paths.dofile('loadModel.lua') --load the model

--------------------------------------------------------
--iterate training:

print '==> training!'

local train = require 'train'
local test  = require 'test'
local trainConf, model, loss
while epoch < opt.maxepoch + 1 do
    --tain on train set
    trainConf, model, loss = train(data.trainData, opt.dataClasses, epoch)

    if (trainError < bestTrainError) then 
        bestTrainError = trainError 
        saveModel(filenameBestTrain, model)  
    end

    --test on val set
    test(data.testData, opt.dataClasses, epoch, trainConf, model, loss )

    local traininfo = {
        lastepoch = epoch,
        lastbestTrainError = bestTrainError,
        lastbestTestError = bestTestError,
        lastbestunion = bestunion
    }

    --saving models and optimstates
    saveModel(filenameLast, model, true)  
    torch.save(opt.save .. '/optimState-last.t7', optimState)
    torch.save(opt.save..'/traininfo-last.t7', traininfo)
    if (epoch % 20 == 0) then		--save snapshot every 20 epochs
        saveModel(paths.concat(opt.save, 'models/model-'..epoch..'.net'), model)  
        torch.save(opt.save .. '/optimStates/optimState-'..epoch..'.t7', optimState)
        torch.save(opt.save..'/traininfos/traininfo-'..epoch..'.t7', traininfo)
    end

    trainConf = nil
    collectgarbage()
    epoch = epoch + 1
end

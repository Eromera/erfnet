-- Eduardo Romera,
-- May 2017.
----------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
    local opt = lapp [[

    Command line options:
    -m, --model       (default 'erfnet_pretrained')      name of network model to profile
    -r, --ratio       (default 0.5) ratio from cityscapes original res (2048x1024)
    --mode				(default 'val')	'val' vs 'test', to decide which subset to produce output
    --subtractmean								use if mean was subtracted during training
    --dataPath			(default '../../datasets/cityscapes/')	path to cityscapes folder
    --saveFolder		(default './save_results/')	folder to save results
    ]]

    return opt
end

opt = opts.parse(arg)
local ratio = opt.ratio
local mode = opt.mode

-------------------

-- Torch packages
require 'image'
require 'cunn'
require 'cudnn'
require 'io'

cudnn.benchmark = true

require 'cityscapesColorMaps'

torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(1)
print("GPU # " .. cutorch.getDevice() .. " selected")

local modelpath = '../trained_models/' .. opt.model .. '.net'
assert(paths.filep(modelpath), 'Model not present at ' .. modelpath)
print("Loading model from: " .. modelpath)

nn.DataParallelTable.deserializeNGPUs = cutorch.getDeviceCount() 
local model = torch.load(modelpath)

--print (model)

local testdir = opt.dataPath .. 'leftImg8bit/' .. mode .. '/'
assert(paths.dirp(testdir), 'Dataset folder does not exist' .. testdir)
local resultsdir = opt.saveFolder .. opt.model .. '/' .. mode .. '/'
local c = 1
f = io.popen('ls ' .. testdir)
for dir in f:lines() do 
    local dpath = testdir .. dir .. '/'
    local dresultspath = resultsdir .. dir .. '/'   

    paths.mkdir(dresultspath)
    f2 = io.popen('ls ' .. dpath)
    for file in f2:lines() do
        --read image and scale to ratio (0.5)
        local imgPath = dpath .. file  
        local img = image.load(imgPath)
        if (ratio ~= 1) then 
            img = image.scale(img, 2048*ratio, 1024*ratio)
        end

        --prepare data to forward
        local inputImg = torch.Tensor(1, 3, img:size(2), img:size(3))
        inputImg[1] = img
        if opt.subtractmean then
            for i=1,3 do inputImg[{{},i}]:csub(cityscapesMean[i]) end	--added for subtractmean models
        end
        inputImgGPU = inputImgGPU or torch.CudaTensor(inputImg:size())
        inputImgGPU:copy(inputImg)
        inputImg = inputImgGPU

        output = model:forward(inputImg)

        --get max of forward:
        _, winners = output:squeeze():max(1)	--mode including unlabelled
        --_, winners = output:squeeze():narrow(1,2,19):max(1)	----mode not including unlabelled
        --winners = winners + 1

        --convert ids (train) to labelIds
        winners = winners:float()
        local winners_labelIds = torch.Tensor(1, img:size(2), img:size(3))
        winners_labelIds = winners:apply(function(x) return reverseClassMap[x][1] end)

        local winners_labelIds_full = winners_labelIds
        if (ratio ~= 1) then 	--scale result size to 2048x1024 (original cityscapes size)
            winners_labelIds_full = image.scale(winners_labelIds, 2048, 1024, 'simple')
        end

        --save result in folder
        local file_aux = string.gsub(file, "leftImg8bit", "result")
        local resultsFilePath = dresultspath .. file_aux
        image.save(resultsFilePath, winners_labelIds_full:byte())

        print ('Saved image ' .. c .. ': ' .. file)

        c = c + 1
    end
end




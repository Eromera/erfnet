-- Eduardo Romera,
-- May 2017.
----------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
    local opt = lapp [[

    Command line options:
    -m, --model       (default 'erfnet_pretrained')		Name of network model to evaluate
    -r, --ratio       (default 0.5) 							Ratio relative to cityscapes original res (2048x1024)
    --mode				(default 'val')						'val' or 'train' subsets (ground-truth is needed)
    --subtractmean													Use if mean was subtracted during training
    -o, --output			(default 'confussionMatrix.txt')	Name of the output file to save conf matrix
    --dataPath			(default '../../datasets/cityscapes/')	path to cityscapes folder
    ]]

    return opt
end

-------------------

opt = opts.parse(arg)
local ratio = opt.ratio
local mode = opt.mode

local classes = {'Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
	             'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
	             'Sky', 'Person', 'Rider', 'Car', 'Truck',
	             'Bus', 'Train', 'Motorcycle', 'Bicycle'}
local conClasses = {'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
	                'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain',
	                'Sky', 'Person', 'Rider','Car', 'Truck',
	                'Bus', 'Train', 'Motorcycle', 'Bicycle'} -- 19 classes

-- Torch packages
require 'image'
require 'cunn'
require 'cudnn'
require 'io'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

local teconfusion = optim.ConfusionMatrix(conClasses)

print '==> remapping classes'

require 'cityscapesColorMaps'

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
local c = 1
f = io.popen('ls ' .. testdir)
for dir in f:lines() do 
    local dpath = testdir .. dir .. '/'

    f2 = io.popen('ls ' .. dpath)
    for file in f2:lines() do

        --read image and scale to 0.5
        local imgPath = dpath .. file  
        local img = image.scale(image.load(imgPath), 2048*ratio, 1024*ratio)

        -- GROUND TRUTH IMAGES
        imgPathGt = string.gsub(imgPath, "leftImg8bit", "gtFine")
        imgPathGt = string.gsub(imgPathGt, ".png", "_labelIds.png")
        local gtImg = image.scale(image.load(imgPathGt,1,'byte'), 2048*ratio, 1024*ratio,'simple')
        gtImg=nn.utils.addSingletonDimension(gtImg)
        gtImg:apply(function(x) return classMap[x][1] end)

        --prepare data to forward
        local inputImg = torch.Tensor(1, 3, img:size(2), img:size(3))
        inputImg[1] = img
        if opt.subtractmean then
            for i=1,3 do inputImg[{{},i}]:csub(cityscapesMean[i]) end	--added for subtractmean models
        end
        inputImgGPU = inputImgGPU or torch.CudaTensor(inputImg:size())
        inputImgGPU:copy(inputImg)
        --inputImg = inputImgGPU

        output = model:forward(inputImgGPU)

        local y = output:transpose(2, 4):transpose(2, 3)
        y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #classes)
        local _, predictions = y:max(2)
        predictions = predictions:view(-1)
        local k = gtImg:view(-1)
        if conClasses then k = k - 1 end
        --k = k:float():apply(function(x) if (x==0) then return -1 end end)	--not necessary, only counted those >0
        teconfusion:batchAdd(predictions, k)

        print ('Processed image ' .. c .. ': ' .. file)
        c = c + 1
    end
end


print (teconfusion)

local file = io.open(opt.output, 'w')
file:write(tostring(teconfusion))
file:close()


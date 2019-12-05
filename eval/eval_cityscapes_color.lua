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
    --mode				(default 'val')	'val', 'test', 'train' or 'demoVideo', to decide which subset to produce output
    --subtractmean								use if mean was subtracted during training
    --dataPath			(default '../../datasets/cityscapes/')	path to cityscapes folder
    --save									flag to activate to save result in files
    --saveFolder		(default './save_color/')	folder to save results
 ]]

    return opt
end

opt = opts.parse(arg)
local ratio = opt.ratio
local mode = opt.mode

-- Torch packages
require 'image'
require 'imgraph'
require 'qtwidget'
require 'cunn'
require 'cudnn'
require 'io'

torch.setdefaulttensortype('torch.FloatTensor')

function pause ()
    print("Press any key to continue.")
    io.flush()
    io.read()
end

print '==> remapping classes'

require 'cityscapesColorMaps'

local colormap = imgraph.colormap(trainIdColors)

cutorch.setDevice(1)
print("GPU # " .. cutorch.getDevice() .. " selected")

local modelpath  = '../trained_models/' .. opt.model .. '.net'
assert(paths.filep(modelpath), 'Model not present at ' .. modelpath)
print("Loading model from: " .. modelpath)

nn.DataParallelTable.deserializeNGPUs = cutorch.getDeviceCount() 
local model = torch.load(modelpath)

--print (model)

local testdir = opt.dataPath .. 'leftImg8bit/' .. mode .. '/'
assert(paths.dirp(testdir), 'Dataset folder does not exist' .. testdir)
local resultsdir = opt.saveFolder .. opt.model .. '/' .. mode .. '/'
local c = 1

print (testdir)

f = io.popen('ls ' .. testdir)
for dir in f:lines() do 
    local dpath = testdir .. dir .. '/'
    local dresultspath = resultsdir .. dir .. '/'   

    if (opt.save) then 
        paths.mkdir(dresultspath)	--create folder for each city
    end

    f2 = io.popen('ls ' .. dpath)
    for file in f2:lines() do

        --read image and scale to 0.5
        local imgPath = dpath .. file  
        local img = image.scale(image.load(imgPath), 2048*ratio, 1024*ratio)

        -- GROUND TRUTH IMAGES
        local gtImg
        if (mode == 'val' or mode == 'train') then
            imgPathGt = string.gsub(imgPath, "leftImg8bit", "gtFine")
            imgPathGt = string.gsub(imgPathGt, ".png", "_labelIds.png")
            gtImg = image.scale(image.load(imgPathGt,1,'byte'), 2048*ratio, 1024*ratio,'simple')
            --gtImg=nn.utils.addSingletonDimension(gtImg)
            gtImg:apply(function(x) return classMap[x][1] end)
            gtImg = imgraph.colorize(gtImg:float(), colormap)
            --winqt3 = image.display{image=gtImg, win=winqt3}
        end

        --prepare data to forward
        local inputImg = torch.Tensor(1, 3, img:size(2), img:size(3))
        inputImg[1] = img
        if opt.subtractmean then
            for i=1,3 do inputImg[{{},i}]:csub(cityscapesMean[i]) end	--added for subtractmean models
        end
        inputImgGPU = inputImgGPU or torch.CudaTensor(inputImg:size())
        inputImgGPU:copy(inputImg)
        --inputImg = inputImgGPU

        --winqt1 = image.display{image=inputImg, win=winqt1}

        output = model:forward(inputImgGPU)

        --get max of forward
        _, winners = output:squeeze():max(1)
        --_, winners = output:squeeze():narrow(1,2,19):max(1)	--not include unlabelled
        --winners = winners + 1

        -- colorize classes
        winners_labelRGB = imgraph.colorize(winners:float(), colormap)


        --winqt2 = image.display{image=winners_labelRGB, win=winqt2}

        --save result in folder
        if (opt.save) then
            local file_aux = string.gsub(file, "leftImg8bit", "result")
            local resultsFilePath = dresultspath .. file_aux
            image.save(resultsFilePath, winners_labelRGB:byte())
        end

        print ('Processed image ' .. c .. ': ' .. file)

        inputImg = inputImg*255		--pass from float to uint for compatibility with labels

        winners_labelRGB = image.scale(winners_labelRGB, 512,256,'simple')
        inputImg = image.scale(inputImg:squeeze(), 512,256,'simple')
        if (mode == 'val' or mode == 'train') then
            gtImg = image.scale(gtImg, 512,256,'simple')
            imageToShow = inputImg:cat(gtImg,3):cat(winners_labelRGB,3)
        else
            imageToShow = inputImg:cat(winners_labelRGB,3)
        end
        winqt4 = image.display{image=imageToShow, win=winqt4}

        c = c + 1
        pause()
    end
end




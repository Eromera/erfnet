-- Eduardo Romera,
-- May 2017.
----------------------------------------------------------------------

-- Torch packages
require 'image'
require 'cunn'
require 'cudnn'
require 'cutorch'

local opts = {}
lapp = require 'pl.lapp'
function opts.parse(arg)
    local opt = lapp [[

    Command line options:
    -m, --model       (default 'erfnet_pretrained')      name of network model to profile
    -p, --platform    (default cudaHalf)     Select profiling platform (cpu|cuda|cudaHalf)
    -r, --res         (default 1x3x512x1024) Input image resolution Channel x Width x Height
    ]]
    return opt
end
opt = opts.parse(arg)
print (opt.model)
print(opt.res)
print(opt.platform)

--torch.setdefaulttensortype('torch.FloatTensor')
--cutorch.setDevice(0)

--cudnn.faster = true
cudnn.benchmark = true

local timer = torch.Timer()      -- whole loop
local totalTime

--sys.sleep(2)

modelpath = '../trained_models/' .. opt.model .. '.net'
assert(paths.filep(modelpath), 'Model not present at ' .. modelpath)
model = torch.load(modelpath)
if torch.typename(model) == 'nn.DataParallelTable' then model = model:get(1) end


--Remove batch-normalization and dropout layers to evaluate forward time, as BN layers can be in fact absorbed by the conv layers by manipulating its weights and biases (https://github.com/e-lab/torch-toolbox/tree/master/BN-absorber)
model:apply(function(module)
    if module.modules then
        for i,submodule in ipairs(module.modules) do
            if torch.typename(submodule):match('cudnn.SpatialBatchNormalization') or 
                torch.typename(submodule):match('nn.SpatialBatchNormalization')	then
                module:remove(i)
            end
        end
        for i,submodule in ipairs(module.modules) do
            if torch.typename(submodule):match('nn.SpatialDropout')	then
                module:remove(i)
            end
        end
    end
end)


if (opt.platform == 'cuda') then
    model:cuda()
elseif (opt.platform == 'cudaHalf') then
    model:cudaHalf()
end


local iBatch, iChannel, iWidth, iHeight = string.match(opt.res, '(%d+)x(%d+)x(%d+)x(%d+)')
iBatch = tonumber(iBatch)
iChannel = tonumber(iChannel)
iWidth = tonumber(iWidth)
iHeight = tonumber(iHeight)

local scaledImg

while(1) do
    --local scaledImg = image.scale(image.load('test.png'),640,360)
    --scaledImg = nn.utils.addSingletonDimension(scaledImg)

    --print (scaledImg:size())
    --scaledImg = torch.Tensor(iBatch, iChannel, iHeight, iWidth)
    --scaledImg:apply(function() return torch.random(0, 255) end)
    --local scaledImgGPU = scaledImgGPU or torch.CudaHalfTensor(scaledImg:size())
    --scaledImgGPU:copy(scaledImg)
    --scaledImg = scaledImgGPU

    if (opt.platform == 'cuda') then
        scaledImg = torch.CudaTensor(iBatch, iChannel, iHeight, iWidth)
    elseif (opt.platform == 'cudaHalf') then
        scaledImg = torch.CudaHalfTensor(iBatch, iChannel, iHeight, iWidth)
    end

    cutorch.synchronize()
    timer:reset()

    output = model:forward(scaledImg)

    cutorch.synchronize()

    totalTime = timer:time().real
    print(totalTime)
    collectgarbage()
end

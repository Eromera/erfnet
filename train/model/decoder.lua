----------------------------------------------------------------------
-- Eduardo Romera,
-- May 2017.
-- Define Decoder Model and call loss function to return trainable network
----------------------------------------------------------------------

require 'torch'   -- torch
torch.setdefaulttensortype('torch.FloatTensor')
local classes = opt.dataClasses

--Model variables:
require './residual_modules'
local prelus=false	--Using ReLU (instead of PReLU) is faster

----------------------------------------------------------------------
print '==> construct model (load encoder + add decoder)'

-- load encoder CNN:
nn.DataParallelTable.deserializeNGPUs = 1
model = torch.load(opt.CNNEncoder)
if torch.typename(model) == 'nn.DataParallelTable' then model = model:clearState():get(1) end
model:remove(#model.modules) -- remove the last layer (classifier)
--nn.DataParallelTable.deserializeNGPUs = 0
model:float()

model:add( upsampler (128, 64))         								-- 1/4 size
model:add( non_bt_1D (64, 64,3,0,prelus,false))
model:add( non_bt_1D (64, 64,3,0,prelus,false))
model:add( upsampler (64, 16))          								-- 1/2 size
model:add( non_bt_1D (16, 16,3,0,prelus,false))
model:add( non_bt_1D (16, 16,3,0,prelus,false))
model:add( cudnn.SpatialFullConvolution(16, #classes, 2, 2, 2, 2))	--original size


----------------------------------------------------------------------
print '==> here is the model:'
--print(model)

normvalue = 1.10	--Value used in the LOSS for class balancing

-- return package:
return {
    model = model2multigpu(model),
    loss = require './loss_general',
}


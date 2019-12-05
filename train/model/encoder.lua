----------------------------------------------------------------------
-- Eduardo Romera,
-- May 2017.
-- Define encoder Model and call loss function to return trainable network
----------------------------------------------------------------------

require 'torch'   -- torch
torch.setdefaulttensortype('torch.FloatTensor')
local classes = opt.dataClasses

--Model variables:
require './residual_modules'
local prelus=false	--Using ReLU (instead of PReLU) is faster
local drop=0.3			--dropout

----------------------------------------------------------------------
print '==> construct model'
local model = nn.Sequential()

model:add( downsampler (3,16,3,0,false))            -- 1/2 size
model:add( downsampler (16,64,3,drop/10.0,false))   -- 1/4 size
                  
for i = 1,5 do
   model:add( non_bt_1D (64, 64, 3, drop/10.0, prelus, false))
end

model:add( downsampler (64,128,3,drop,false))   -- 1/8 size
for i = 1,2 do
    model:add( non_bt_1D (128, 128, 3, drop, prelus, 2))	 --dilated 2
    model:add( non_bt_1D (128, 128, 3, drop, prelus, 4))   --dilated 4
    model:add( non_bt_1D (128, 128, 3, drop, prelus, 8))   --dilated 8
    model:add( non_bt_1D (128, 128, 3, drop, prelus, 16))  --dilated 16
end
model:add(cudnn.SpatialConvolution(128, #classes, 1, 1))

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

normvalue = 1.2	--Value used in the LOSS for class balancing


-- return package:
return {
    model = model2multigpu(model),
    loss = require './loss_general',
}


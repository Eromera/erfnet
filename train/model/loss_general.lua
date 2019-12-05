----------------------------------------------------------------------
-- Eduardo Romera,
-- May 2017.
-- Define loss function with class balancing
----------------------------------------------------------------------

require 'torch'   -- torch
torch.setdefaulttensortype('torch.FloatTensor')

print '==> define parameters'
local histClasses = opt.datahistClasses
local classes = opt.dataClasses
local conClasses = opt.dataconClasses

print('defining loss function:')

--Class Balancing:
local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
    if histClasses[i] < 1 or i == 1 then --ignore unlabeled
        classWeights[i] = 0
    else
        classWeights[i] = 1 / (torch.log(normvalue + normHist[i]))		--normvalue has been defined in model file
    end
    print (classWeights[i])
end

-- Loss: NLL
loss = cudnn.SpatialCrossEntropyCriterion(classWeights) 

loss:cuda()

return loss

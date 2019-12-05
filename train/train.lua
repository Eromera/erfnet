----------------------------------------------------------------------
-- This script
--   + constructs mini-batches on the fly
--   + computes model error
--   + optimizes the error using several optmization
--     methods: SGD, L-BFGS, ADAM.
--
-- Originally written by : Abhishek Chaurasia, Eugenio Culurcielo
--
-- Modified by Eduardo Romera (donkeys, getBatch...)
-- May 2017.
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'getBatch'

local Threads = require 'threads'
local Queue = require 'threads.queue'
Threads.serialization('threads.sharedserialize')

torch.setdefaulttensortype('torch.FloatTensor')

local loss = t.loss

 ----------------------------------------------------------------------

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w, dE_dw
local model = t.model
print '==> flattening model parameters'
w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print '==> defining training procedure'

local confusion
if (opt.noConfusion == 'all') then
    if opt.dataconClasses then
        print('Class \'Unlabeled\' is ignored in confusion matrix')
        confusion = optim.ConfusionMatrix(opt.dataconClasses)
    else
        confusion = optim.ConfusionMatrix(opt.dataClasses)
    end
else
    confusion = nil
end

----------------------------------------------------------------------
print '==> allocating minibatch memory'

local function train(trainData, classes, epoch)

    if (epoch ~= 0 and epoch == deactivateWDepoch) then
        opt.weightDecay = 0
    end


    if (epoch % opt.lrDecayEvery == 0) then 
        optimState.learningRate = optimState.learningRate * opt.lrDecayManual 
    end

    opt.learningRate = optimState.learningRate   --update just for visualization purposes

    -- local vars
    local time = sys.clock()

    -- total loss error
    local err
    local totalerr = 0

    -- shuffle at each epoch
    local shuffle = torch.randperm(trainData:size())

    local batchesx = {}	--list to store loaded batches, each element contains a batch
    local batchesyt = {}

    --if loadmode==1 loads data from threads (opt.nDonkeys)
    local donkeys
    if (opt.loadMode == 1 and opt.nDonkeys > 0) then
    local optUp = opt --upvalue for global variable opt
    local classMapUp = classMap -- upvalue for global variable classmap
    donkeys = Threads(
        opt.nDonkeys,
        Threads.safe(function()		--Threads.safe is necessary to avoid deadlock problem caused by fork with require, otherwise sometimes it hangs while doing require 'image', --https://github.com/torch/image/issues/195
            torch.setdefaulttensortype('torch.FloatTensor')
            require 'getBatch'
        end),
        function(threadid)
            --print(string.format('Starting donkey with id: %d', threadid))
            opt = optUp -- pass to all donkeys via upvalue
            classMap = classMapUp	--upvalue 
            nextBatch = 1 + (threadid-1) * opt.batchSize --value local to each thread (know what batch to load) 
        end
        );

        donkeys:specific(true)	--to keep order we set which thread does which job
        donkeys:synchronize()
    end

    local function getFromThreads()
        --this gets called when we add job
        if (nextBatch + opt.batchSize-1 > trainData:size()) then
            return __threadid, nil, nil
        end

        local resx, resyt = getBatchTrain(nextBatch, trainData, shuffle)
        collectgarbage()	--seems to be the only position to effectively retain the memory going to the moon
        nextBatch = nextBatch + (opt.batchSize * opt.nDonkeys)	--local variable of each thread
        return __threadid, resx, resyt 
    end

    local function pushResult(idx, resx, resyt)
        --this gets called after the job has been completed (dojob or synchronize)
        batchesx[idx] = resx
        batchesyt[idx] = resyt
    end

    if (opt.loadMode == 1 and opt.nDonkeys > 0) then
        for i = 1, opt.nDonkeys do
            donkeys:addjob(i, getFromThreads, pushResult)
        end
    end

    local iCurrentDonkey = 1

    model:training()
    -- do one epoch
    print("==> Training: epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1, trainData:size(), opt.batchSize do	
        -- disp progress
        xlua.progress(t, trainData:size())

        -- batch fits?
        if (t + opt.batchSize - 1) > trainData:size() then
            break
        end

        -- create mini batch
        local x,yt 
        if (opt.loadMode == 0) then
            x,yt = getMiniBatch(t, trainData, shuffle)
        else
            while (batchesx[iCurrentDonkey] == nil or batchesyt[iCurrentDonkey] == nil) do
                donkeys:dojob() --donkeys:synchronize()
            end

            x = batchesx[iCurrentDonkey]
            yt = batchesyt[iCurrentDonkey]		

            batchesx[iCurrentDonkey] = nil
            batchesyt[iCurrentDonkey] = nil

            --add new job to load another batch (if t>trsize the job will do nothing)
            donkeys:addjob(iCurrentDonkey, getFromThreads, pushResult)

            --increment "i" cycling between 1 and opt.nDonkeys, for next load:
            iCurrentDonkey = (iCurrentDonkey) % opt.nDonkeys + 1

        end

        x = x:cuda()
        yt = yt:cuda()

        local y

        -- create closure to evaluate f(X) and df/dX
        local eval_E = function(w)
            -- reset gradients
            model:zeroGradParameters()

            -- evaluate function for complete mini batch
            y = model:forward(x)

            --[[--DEBUG EACH FEATURE MAP:
            local input = x
            for h=1,model:size() do
            input = model:get(h):forward(input)
            print ('Module: ' .. h)
            print (input:size())
            end
            y = input
            --]]	

            -- estimate df/dW
            err = loss:forward(y,yt)            -- updateOutput

            local dE_dy = loss:backward(y,yt)   -- updateGradInput

            model:backward(x,dE_dy)
            -- Don't add this to err, so models with different WD
            -- settings can be easily compared. optim functions
            -- care only about the gradient anyway (adam/rmsprop)
            dE_dw:add(opt.weightDecay, w)

            -- return f and df/dX
            return err, dE_dw
        end

        -- optimize on current mini-batch
        local _, errt = optim.adam(eval_E, w, optimState)

        if opt.printNorm == true then
            local norm = opt.learningRate * dE_dw:norm() / w:norm()
            print(string.format('train err: %f, norm : %f epoch: %d   lr: %f  ', errt[1], norm, epoch, opt.learningRate))
        end
        -- update confusion
        if opt.noConfusion == 'all' then
            model:evaluate()
            local y2 = y:clone():transpose(2, 4):transpose(2, 3) 
            y2 = y2:reshape(y2:numel()/y2:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
            local _, predictions = y2:max(2)
            predictions = predictions:view(-1)
            local k = yt:view(-1)
            if opt.dataconClasses then k = k - 1 end
            confusion:batchAdd(predictions, k)
            model:training()
        end

        totalerr = totalerr + err
        collectgarbage()
    end

    if opt.loadMode == 1 then
        donkeys:terminate()
    end

    -- time taken
    time = sys.clock() - time
    time = time / trainData:size()
    totalerr = totalerr / (trainData:size()/opt.batchSize) -- average error in train dataset
    local optimclr = optimState.learningRate / (1 + optimState.t*optimState.learningRateDecay)

    print(string.format("==> time to learn 1 sample: %.2f ms // it: %d // lr: %f // Train-err: \27[36m%f%%", time*1000, optimState.t, optimclr, totalerr))

    trainError = totalerr

    model:clearState()	--clear intermediate states for saving space when saving model

    collectgarbage()
    return confusion, model, loss
end

-- Export:
return train

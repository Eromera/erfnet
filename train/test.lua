----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data.
--
-- Originally written by : Abhishek Chaurasia, Eugenio Culurcielo
--
-- Modified by Eduardo Romera (donkeys, getBatch, print IoU...)
-- May 2017.
----------------------------------------------------------------------
----------------------------------------------------------------------

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'getBatch'

torch.setdefaulttensortype('torch.FloatTensor')

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

----------------------------------------------------------------------
-- Logger:
coTotalLogger = optim.Logger(paths.concat(opt.save, 'confusionTotal.log'))
coAveraLogger = optim.Logger(paths.concat(opt.save, 'confusionAvera.log'))
coUnionLogger = optim.Logger(paths.concat(opt.save, 'confusionUnion.log'))

print '==> defining test procedure'
local teconfusion, filename

if opt.dataconClasses then
    teconfusion = optim.ConfusionMatrix(opt.dataconClasses)
else
    teconfusion = optim.ConfusionMatrix(opt.dataClasses)
end

-- test function
function test(testData, classes, epoch, trainConf, model, loss )
    ----------------------------------------------------------------------
    -- local vars
    local time = sys.clock()
    -- total loss error
    local err = 0
    local totalerr = 0

    -- This matrix records the current confusion across classes
    model:evaluate()

    -- Batch test:
    local shuffle = torch.range(1,testData:size())	--dummy shuffle for getBatch

    local batchesx = {}	--list to store loaded batches, each element contains a batch
    local batchesyt = {}

    --loadmode==1 loads data from threads (opt.nDonkeys)
    local donkeys
    if opt.nDonkeys > 0 then
        local optUp = opt --upvalue for global variable opt
        local classMapUp = classMap -- upvalue for global variable classmap
        donkeys = Threads(
            opt.nDonkeys,
            Threads.safe(function() --Threads.safe is necessary to avoid deadlock problem caused by fork with require, otherwise sometimes it hangs while doing require 'image', --https://github.com/torch/image/issues/195
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
        if (nextBatch + opt.batchSize-1 > testData:size()) then
            return __threadid, nil, nil
        end

        local resx, resyt = getBatchTest(nextBatch, testData, shuffle)

        collectgarbage()	
        nextBatch = nextBatch + (opt.batchSize * opt.nDonkeys)	
        return __threadid, resx, resyt 
    end

    local function pushResult(idx, resx, resyt)
        --this gets called after the job has been completed (dojob or synchronize)
        batchesx[idx] = resx
        batchesyt[idx] = resyt
    end

    if (opt.nDonkeys > 0) then
        for i = 1, opt.nDonkeys do
            donkeys:addjob(i, getFromThreads, pushResult)
        end
    end

    local iCurrentDonkey = 1

    -- test over test data
    print('==> Testing:')
    for t = 1, testData:size(), opt.batchSize do
        -- disp progress
        xlua.progress(t, testData:size())

        -- batch fits?
        if (t + opt.batchSize - 1) > testData:size() then
            break
        end

        -- create mini batch
        -- create mini batch
        local x,yt 
        if (opt.nDonkeys == 0) then
            x,yt = getMiniBatch(t, testData, shuffle)
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

        -- test sample
        local y = model:forward(x)

        local err = loss:forward(y,yt)
        if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
            y = y:transpose(2, 4):transpose(2, 3)
            y = y:reshape(y:numel()/y:size(4), #classes):sub(1, -1, 2, #opt.dataClasses)
            local _, predictions = y:max(2)
            predictions = predictions:view(-1)
            local k = yt:view(-1)
            if opt.dataconClasses then k = k - 1 end
            teconfusion:batchAdd(predictions, k)
        end

        totalerr = totalerr + err
        collectgarbage()
    end

    if opt.loadMode == 1 then
        donkeys:terminate()
    end

    -- timing
    time = sys.clock() - time
    time = time / testData:size()
    -- print average error in train dataset
    totalerr = totalerr / (testData:size() / opt.batchSize)
    --print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms' .. '// Test Error: ' .. totalerr)
    print(string.format("==> time to test 1 sample: %.2f ms // Test-err: \27[36m%f%%", time*1000, totalerr))

    --print(cutorch.getMemoryUsage(1))
    --model = model:clearState()

    if totalerr < bestTestError then
        saveModel(filenameBest, model)	
        -- update to min error:
        if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
            --print('==> saving confusion to '..filename)
            torch.save(paths.concat(opt.save,'confusionData/confusion-best.t7'),teconfusion)

            filename = paths.concat(opt.save, 'confusionMatrix-best.txt')
            --print('==> saving confusion matrix to ' .. filename)
            local file = io.open(filename, 'w')
            file:write("--------------------------------------------------------------------------------\n")
            file:write("Training:\n")
            file:write("================================================================================\n")
            file:write(tostring(trainConf))
            file:write("\n--------------------------------------------------------------------------------\n")
            file:write("Testing:\n")
            file:write("================================================================================\n")
            file:write(tostring(teconfusion))
            file:write("\n--------------------------------------------------------------------------------")
            file:close()

            coTotalLogger:add{['confusion total accuracy'] = teconfusion.totalValid * 100.0 }
            coAveraLogger:add{['confusion average accuracy'] = teconfusion.averageValid * 100.0 }
            coUnionLogger:add{['confusion union accuracy'] = teconfusion.averageUnionValid*100.0 }
        end
        filename = paths.concat(opt.save, 'best-number.txt')
        local file = io.open(filename, 'w')
        file:write("----------------------------------------\n")
        file:write("Best test error: ")
        file:write(tostring(totalerr))
        file:write(", Test Class IoU: ")
        if (opt.noConfusion == 'tes' or opt.noConfusion == 'all') then
            file:write(tostring(teconfusion.averageUnionValid  * 100.0)) else
            file:write('not available') end		
            file:write(", in epoch: ")
            file:write(tostring(epoch))
            file:write("\n==> time to test 1 sample = " .. tostring(time*1000) .. 'ms')
            file:write("\n----------------------------------------\n")
            file:close()
            bestTestError = totalerr
        end

    if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
        -- update to min error:
        filename = paths.concat(opt.save, 'confusionMatrixes/confusionMatrix-' .. epoch .. '.txt')
        --print('==> saving confusion matrix to ' .. filename)
        local file = io.open(filename, 'w')
        file:write("--------------------------------------------------------------------------------\n")
        file:write("Training:\n")
        file:write("================================================================================\n")
        file:write(tostring(trainConf))
        file:write("\n--------------------------------------------------------------------------------\n")
        file:write("Testing:\n")
        file:write("================================================================================\n")
        file:write(tostring(teconfusion))
        file:write("\n--------------------------------------------------------------------------------")
        file:close()


        --print('==> saving test confusion object to '..filename)
        filename = paths.concat(opt.save,'confusionData/confusion-last.t7')  --changed now it doesnt save every epoch to save space, only last, best and bestiou
        torch.save(filename,teconfusion)

        if teconfusion.averageUnionValid > bestunion then
            saveModel(filenameBestiou, model) --:clearState()
            -- update to min error:
            if opt.noConfusion == 'tes' or opt.noConfusion == 'all' then
                --print('==> saving confusion to '..filename)
                filename = paths.concat(opt.save,'confusionData/confusion-bestiou.t7')
                torch.save(filename,teconfusion)

                filename = paths.concat(opt.save, 'confusionMatrix-bestiou.txt')
                --print('==> saving confusion matrix to ' .. filename)
                local file = io.open(filename, 'w')
                file:write("--------------------------------------------------------------------------------\n")
                file:write("Training:\n")
                file:write("================================================================================\n")
                file:write(tostring(trainConf))
                file:write("\n--------------------------------------------------------------------------------\n")
                file:write("Testing:\n")
                file:write("================================================================================\n")
                file:write(tostring(teconfusion))
                file:write("\n--------------------------------------------------------------------------------")
                file:close()
            end
            filename = paths.concat(opt.save, 'best-iou.txt')
            local file = io.open(filename, 'w')
            file:write("----------------------------------------\n")
            file:write("Best test error: ")
            file:write(tostring(totalerr))
            file:write(", Test Class IoU: ")
            if (opt.noConfusion == 'tes' or opt.noConfusion == 'all') then
            file:write(tostring(teconfusion.averageUnionValid  * 100.0)) else
            file:write('not available') end	
            file:write(", in epoch: ")
            file:write(tostring(epoch))
            file:write("\n==> time to test 1 sample = " .. tostring(time*1000) .. 'ms')
            file:write("\n----------------------------------------\n")
            file:close()
            bestunion = teconfusion.averageUnionValid
        end

    end


    --NOTE: for some reason trainConf & teconfusion values are 0 before you call toString(..), so this piece of code must be here at the end
    --print('==> Adding data to automated_log')
    filename = paths.concat(opt.save, 'automated_log.txt')
    local file2 
    if epoch==1 then 
        file2 = io.open(filename, 'w')
        file2:write("Epoch\t\tTrain-err\t\tTest-err\t\tTrain-acc\t\tTest-acc\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate") 
    else
        file2 = io.open(filename, 'a')
    end
    local val1=0
    local val2=0
    local val3=0 
    local val4=0
    if opt.noConfusion == 'all' then
        val1= (trainConf.totalValid~=nil and trainConf.totalValid * 100.0 or -1)
        val2= (teconfusion.totalValid~=nil and teconfusion.totalValid * 100.0 or -1)
        val3= (trainConf.averageUnionValid~=nil and trainConf.averageUnionValid * 100.0 or -1)
        val4= (teconfusion.averageUnionValid~=nil and teconfusion.averageUnionValid * 100.0 or -1)

        --moved this here
        trainConf:zero()
        teconfusion:zero()
    elseif opt.noConfusion == 'tes' then
        val2= (teconfusion.totalValid~=nil and teconfusion.totalValid * 100.0 or -1)
        val4= (teconfusion.averageUnionValid~=nil and teconfusion.averageUnionValid * 100.0 or -1)

        --moved this here
        teconfusion:zero()
    end
    print (string.format("[train-acc, test-acc, train-IoU, test-IoU]: [\27[34m%.2f%%, \27[33m%.2f%%, \27[36m%.2f%%, \27[31m%.2f%%]", val1, val2, val3, val4))

    file2:write(string.format("\n%d\t\t%.4f\t\t%.4f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.8f", epoch,           trainError, totalerr,val1,val2, val3, val4, opt.learningRate ))
    file2:close()

    print('\n') -- separate epochs

    model:clearState()

    x=nil
    yt=nil

    collectgarbage()
end

-- Export:
return test

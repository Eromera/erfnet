-- Eduardo Romera,
-- May 2017.
-- This code is for loading model from scratch or from snapshots
----------------------------------------------------------------------

t = paths.dofile(opt.model)

--print '==> here is the model:'
--print(t.model)

--Reload training snapshot for Epoch opt.loadEpoch:
if (opt.loadEpoch ~= 0) then  --reload training
    local traininfo = torch.load(opt.save..'/traininfos/traininfo-'..opt.loadEpoch..'.t7' )	
    epoch = traininfo.lastepoch
    bestTrainError = traininfo.lastbestTrainError
    bestTestError = traininfo.lastbestTestError
    bestunion = traininfo.lastbestunion

    --nn.DataParallelTable.deserializeNGPUs = opt.nGPU
    print ("Loading model from epoch " .. opt.loadEpoch)
    t.model = torch.load(opt.save .. '/models/model-' .. opt.loadEpoch .. '.net')
    t.model = model2multigpu(t.model)

    --reload optimstate:
    optimState = torch.load(opt.save .. '/optimStates/optimState-'..opt.loadEpoch..'.t7')

    --update for new epoch
    epoch = opt.loadEpoch + 1
end

--Reload training snapshot for last trained epoch
if (opt.loadLastEpoch) then
    --reload some training info:
    local traininfo = torch.load(opt.save..'/traininfo-last.t7' )	
    epoch = traininfo.lastepoch
    bestTrainError = traininfo.lastbestTrainError
    bestTestError = traininfo.lastbestTestError
    bestunion = traininfo.lastbestunion

    --reload model:
    t.model = torch.load(opt.save .. '/model-last.net')
    t.model = model2multigpu(t.model)	

    --reload optimstate:
    optimState = torch.load(opt.save .. '/optimState-last.t7')

    --update for new epoch
    epoch = epoch + 1	--we saved last epoch, want to restart iterating from last+1

end

return t

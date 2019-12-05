-- Eduardo Romera,
-- May 2017.
-- Some general functions
----------------------------------------------------------------------

--save model function
function saveModel(filename, model, noprint)
    if (noprint ~= true) then
        print('==> saving '.. filename)
    end
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(filename, model:clearState():get(1))
    elseif torch.type(model) == 'nn.Sequential' then
        torch.save(filename, model:clearState())
    else
        error('This saving function only works with Sequential or DataParallelTable modules.')
    end
end

--pause function for Lua (useful for debugging)
function pause ()
    print("Press any key to continue.")
    io.flush()
    io.read()
end


--Make model parallel for multi-gpu training
function model2multigpu (model)
    print(cutorch.getDeviceCount() .. " GPUs being used")
    if cutorch.getDeviceCount() > 1 then
        local gpu_list = {}
        for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpu_list)
        :threads(function()
        local cudnn = require 'cudnn'
        --cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()

        return model
    else 
        return model:cuda()
    end
end

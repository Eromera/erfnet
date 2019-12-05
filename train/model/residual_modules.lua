----------------------------------------------------------------------
-- Eduardo Romera,
-- May 2017.
-- Define residual modules (non_bottleneck_1D...)
----------------------------------------------------------------------

require 'torch'   -- torch
torch.setdefaulttensortype('torch.FloatTensor')

--Notes:
--RELU vs PReLU: cudnn.relu is faster and uses less memory (cudnn in place option), but nn.prelu has a slight gain from the extra parameter that learns the negative slope.

function non_bt_1D (input, output, ksize, dropprob, prelus, dilated)

    local sum = nn.ConcatTable()
    local main = nn.Sequential()
    local other = nn.Sequential()
    sum:add(main):add(other)

    local internal = output
    local pad = (ksize-1) / 2


    main:add(cudnn.SpatialConvolution(input, internal, ksize, 1, 1, 1, pad, 0))   
    main:add(prelus and nn.PReLU(internal) or cudnn.ReLU(true))
    main:add(cudnn.SpatialConvolution(internal, internal, 1, ksize, 1, 1, 0, pad))
    main:add(cudnn.SpatialBatchNormalization(internal, 1e-3))
    main:add(prelus and nn.PReLU(internal) or cudnn.ReLU(true))
    if  not dilated then
        main:add(cudnn.SpatialConvolution(internal, internal, ksize, 1, 1, 1, pad, 0))
        main:add(prelus and nn.PReLU(internal) or cudnn.ReLU(true))
        main:add(cudnn.SpatialConvolution(internal, output, 1, ksize, 1, 1, 0, pad))
    elseif dilated then
        main:add(nn.SpatialDilatedConvolution(internal, internal, ksize, 1, 1, 1, pad*dilated, 0, dilated, 1))
        main:add(prelus and nn.PReLU(internal) or cudnn.ReLU(true))
        main:add(nn.SpatialDilatedConvolution(internal, output, 1, ksize, 1, 1, 0, pad*dilated, 1, dilated))
    else
        assert(false, 'You shouldn\'t be here')
    end
    main:add(cudnn.SpatialBatchNormalization(output, 1e-3))
    if (droppprob ~= 0) then main:add(nn.SpatialDropout(dropprob)) end

    other:add(nn.Identity())

    local modul = nn.Sequential():add(sum):add(nn.CAddTable(true))
    modul:add(prelus and nn.PReLU(output) or cudnn.ReLU(true))

    return modul
end


function downsampler (input, output, ksize, dropprob, prelus)

    local sum = nn.ConcatTable()
    local main = nn.Sequential()
    local other = nn.Sequential()
    sum:add(main):add(other)

    local pad = (ksize-1) / 2

    main:add(cudnn.SpatialConvolution(input, output-input, ksize, ksize, 2, 2, pad, pad)) 

    other:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))    

    local modul = nn.Sequential():add(sum):add(nn.JoinTable(2))
    modul:add(cudnn.SpatialBatchNormalization(output, 1e-3))
    if (droppprob ~= 0) then modul:add(nn.SpatialDropout(dropprob)) end
    modul:add(prelus and nn.PReLU(output) or cudnn.ReLU(true))

    return modul
end

function upsampler (input, output)
    local input_stride = upsample and 2 or 1

    local modul = nn.Sequential()

    modul:add(cudnn.SpatialFullConvolution(input, output, 3, 3, 2, 2, 1, 1, 1, 1))

    modul:add(cudnn.SpatialBatchNormalization(output, 1e-3))
    modul:add(cudnn.ReLU(true))

    return modul
end


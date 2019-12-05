-- Eduardo Romera,
-- May 2017.
-- This code is for parsing arguments/options and defining some global variables
----------------------------------------------------------------------

--load defined options
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

------------------------------------------------------------------------
if (opt.decoder) then 
    print ('DECODER MODE')
    opt.labelHeight = opt.imHeight
    opt.labelWidth = opt.imWidth
else
    print ('ENCODER MODE')
    opt.labelHeight = opt.imHeight * 0.125
    opt.labelWidth = opt.imWidth * 0.125
end

local saveName = opt.imWidth .. 'x' .. opt.imHeight .. '/'
if opt.customSave ~= '' then
    saveName = saveName .. opt.customSave
else
    assert (false, 'ERROR: YOU MUST SPECIFY A NAME FOR THIS TRAINING SAVE FOLDER')
end
opt.save = paths.concat(opt.save, saveName)


if (opt.decoder) then
    if (opt.CNNEncoder=='') then opt.CNNEncoder = opt.save .. '/enc/model-bestiou.net' end
    opt.save = opt.save .. '/dec'
    opt.model = opt.model .. '/decoder.lua'
else
    opt.save = opt.save .. '/enc'
    opt.model = opt.model .. '/encoder.lua'  
end

paths.mkdir(opt.save)
assert(paths.dirp(opt.save), 'Error: save folder incorrect! ' .. opt.save)
print("Folder created at " .. opt.save)

------------------------------------------------------------------------

print 'saving opt as txt and t7'
local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()
torch.save(path.join(opt.save,'opt.t7'),opt)

paths.mkdir(opt.save .. '/confusionData')
assert(paths.dirp(opt.save .. '/confusionData'), 'confusionData/ not created!: ')
paths.mkdir(opt.save .. '/confusionMatrixes')
assert(paths.dirp(opt.save .. '/confusionMatrixes'), 'confusionMatrixes/ not created!: ')
paths.mkdir(opt.save .. '/models')
assert(paths.dirp(opt.save .. '/models'), 'models/ not created!: ')
paths.mkdir(opt.save .. '/optimStates')
assert(paths.dirp(opt.save .. '/optimStates'), 'optimStates/ not created!: ')
paths.mkdir(opt.save .. '/traininfos')
assert(paths.dirp(opt.save .. '/traininfos'), 'traininfos/ not created!: ')

--globals to save models:
filenameLast = paths.concat(opt.save, 'model-last.net')
filenameBestTrain = paths.concat(opt.save, 'model-besttrain.net')
filenameBest = paths.concat(opt.save, 'model-best.net')
filenameBestiou = paths.concat(opt.save, 'model-bestiou.net')

---------------------------------------------------------------------

return opt


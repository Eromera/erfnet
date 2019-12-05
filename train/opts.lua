-- Eduardo Romera,
-- May 2017.
-- Define program options here
----------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
    local opt = lapp [[
    Command line options:

    --datapath              (default ../../datasets/cityscapes/)  dataset location
    --cachepath             (default ../cache/)  cache directory to save the loaded dataset
    --save                  (default ../save/)     main save folder 
    --customSave            (default '')			 this name is used to create a new folder for this specific training

    -r,--learningRate       (default 5e-4)        learning rate 
    -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples) 
    -w,--weightDecay        (default 2e-4)        L2 penalty on the weights 
    -m,--momentum           (default 0.9)         momentum
    -b,--batchSize          (default 10)          batch size 
    --maxepoch              (default 300)         maximum number of training epochs
    --lrDecayEvery          (default 50)          Decay learning rate every X epoch by lrDecayManual
    --lrDecayManual         (default 0.5)         cada "lrDecayEvery", lr = learningRate * lrDecayManual  

    --channels              (default 3)
    --imWidth               (default 1024)        image width   
    --imHeight              (default 512)         image height 
    --smallNet                                    reduce number of classes

    --loadMode					(default 0)				 0=preload all dataset in memory, 1=load dataset images on the iterations
    --nDonkeys					(default 6)				 number of donkeys (threads) to load data

    --model                 (default model/)      Path of the model folder
    --decoder                                     activate when training decoder
    --CNNEncoder            (default '')        pretrained encoder for which you want to train your decoder

    --noConfusion           (default tes) 		    skip: skip confusion matrix calc, all: test+train, tes : test only
    --printNorm                                   For visualize norm factor while training

    --loadEpoch             (default 0)           for loading from specific snapshots
    --loadLastEpoch										 reload directly model-last.net and last training snapshot
    --augment               				          activate to augment data
    --subtractmean											 to subtract dataset mean during train
 ]]	


    return opt
end

return opts

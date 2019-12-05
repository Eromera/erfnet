# Training ERFNet

Torch code for training the model. It loads the dataset and the defined architecture and iterates by training on the train set and testing on the val set.

The train/test codes are based on the ENet Torch code ([ENet-training](https://github.com/e-lab/ENet-training/tree/master/train)), adapted with several custom modifications and tweaks (e.g. snapshot saving, loading dataset with threads...). It is mainly prepared for Cityscapes dataset, but the data-loader code can be easily adapted for other datasets.


## Files/folders and their function:

* [data](data)          : data loaders for loading the dataset
* [model](model)        : model architecture is defined here
* [main.lua](main.lua)    : main file
* [train.lua](train.lua) : iterate through batches and perform forward/backpropagation
* [test.lua](test.lua)  : calculate testing error and save confusion matrices
* [opts.lua](opts.lua)  : contains the user-defined options for the program
* [loadOptions.lua](loadOptions.lua)   : parse arguments/options and define some global variables
* [loadModel.lua](loadModel.lua) : load model and control loading from snapshots
* [getBatch.lua](getBatch.lua)  : called from train/test to get batches and perform data augmentations
* [utils.lua](utils.lua)  : define some general functions

## Options

Please see 'opts.lua' to review all options and default values. The main ones are:

* **--customSave** : name to create folder for saving everything related to this training. 
* **--datapath** : path to Cityscapes directory (which must contain leftImg8bit and gtFine folders).
* **--imWidth** : image width (default 1024).
* **--imHeight** : image height (default 512).
* **-b,--batchSize** : Batch size, split between GPUs.
* **--loadMode** : can be '0' or '1'. Mode '0' (default) will preload all dataset in memory in 8bits (and will save a cache file to accelerate further loadings) so the image batches can be loaded fast from memory during train/test. Mode '1' will load the list of files from 'train.txt' and 'val.txt' and will load the dataset images from disk during each iteration by using threads (specified by option nDonkeys).
* **--decoder**	: use this flag to activate decoder training mode.
* **--CNNEncoder**	: this option can be used in decoder mode to specify the encoder to load (otherwise as default it loads 'enc/model-bestiou.net' from the specified 'customSave' folder).
* **--augment** : use this flag to activate simple augmentations (random translations of 0-2 pixels and horizontal flips).

## Example command for training encoder:

```
th main.lua --customSave erfnet_scratch --datapath /home/datasets/cityscapes/ -b 12 --augment --loadMode 1
```

## Example command for training decoder:

```
th main.lua --customSave erfnet_scratch --datapath /home/datasets/cityscapes/ -b 12 --augment --loadMode 1 --decoder 
th main.lua --customSave erfnet_pretrained --datapath /home/datasets/cityscapes/ -b 12 --augment --loadMode 1 --decoder --CNNEncoder ../trained_models/pretrained_encoder_imagenet.net 
```

## Multi-GPU training:

To control how many GPUs to use during training, please use **CUDA_VISIBLE_DEVICES** before the command. If not specified, the code will use all available GPUs. The first one specified will allocate extra memory for synchronization.

```
CUDA_VISIBLE_DEVICES=0 th main.lua ...			#1 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 th main.lua ...	#4 GPUS
```

# Functions for evaluating/visualizing the network's output

------
## eval_cityscapes_color.lua 

This code can be used to visualize or/and save the colorized output of the network for images in the Cityscapes dataset. Class colors correspond to those used in Cityscapes dataset. 

**Important: 'imgraph' lua package is required** (install with 'luarocks install imgraph').

**Options:** Specify the Cityscapes folder path with '--dataPath' option. Select the cityscapes subset with '--mode' ('val', 'test', 'train' or 'demoVideo'), the code will iterate through that folder and load the subset images. If 'val' or 'train' are selected, ground truth images are loaded and displayed between input image and output. Use the '--save' flag to save results in a folder specified in --saveFolder (default './save_color/').  Option '-m' or '--model' specifies the model to load from folder 'trained_models'.

**Examples:**
```
qlua eval_cityscapes_color.lua
qlua eval_cityscapes_color.lua --mode demoVideo -m erfnet_pretrained
qlua eval_cityscapes_color.lua --mode val --dataPath /home/datasets/citscapes/ --save --saveFolder ./save_color/
```

NOTE: 'qlua' (instead of 'th') is needed for visualization

------
## eval_cityscapes_server.lua 

This code can be used to produce segmentation of the Cityscapes images and convert the output indices to the original 'labelIds' so it can be evaluated using the scripts from Cityscapes dataset (evalPixelLevelSemanticLabeling.py) or uploaded to Cityscapes test server.

**Options:** Specify the Cityscapes folder path with '--dataPath' option. Select the cityscapes subset with '--mode' ('val' or 'test'). Select the folder for saving results with '--saveFolder' option (default './save_results/').  Option '-m' or '--model' specifies the model to load from folder 'trained_models'.

**Examples:**
```
th eval_cityscapes_server.lua
th eval_cityscapes_server.lua --mode test	--dataPath /home/datasets/citscapes/ --saveFolder ./save_results/
```

------
## eval_cityscapes_confussionMatrix.lua 

This code can be used to calculate the confussion matrix for the Cityscapes classes on val or test subsets. The conf matrix is calculated using the 'optim' lua package. 

**Options:** Specify the Cityscapes folder path with '--dataPath' option. Option '-o' or '--output' specifies output file name (default: 'confussionMatrix.txt'). Select the cityscapes subset with '--mode' ('val' or 'test'). Option '-m' or '--model' specifies the model to load from folder 'trained_models'.

**Examples:**
```
th eval_cityscapes_confussionMatrix.lua
th eval_cityscapes_confussionMatrix.lua --mode test --dataPath /home/datasets/citscapes/ -m erfnet_pretrained -o confussionMatrix.txt
```

------
## eval_forwardTime.lua
This function loads a model specified by '-m' and enters a loop to continuously estimate forward pass time (fwt) in the specified resolution. 

**Options:** Option '--platform' specifies the processing mode (cpu | cuda | cudaHalf), where 'cudaHalf' is cuda mode with FP16. Option '--res' specifies the resolution in format batch x channels x height x width (default 1x3x512x1024). Option '-m' or '--model' specifies the model to load from folder 'trained_models'.

**Examples:**
```
th eval_forwardTime.lua
th eval_forwardTime.lua -p cudaHalf --res 1x3x360x640 -m erfnet_pretrained
```

NOTE: Paper values were obtained with a single Titan X (Maxwell) and a Jetson TX1

# ERFNet

This code is a toolbox that uses Torch library for training and evaluating the ERFNet architecture for semantic segmentation. 

## **NEW!! New PyTorch version is available [HERE](https://github.com/Eromera/erfnet_pytorch)**

![Example segmentation](example_segmentation.png?raw=true "Example segmentation")

## Publications

If you use this software in your research, please cite our publications:

**"Efficient ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017. 
**[Best Student Paper Award]**, [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

**"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, Transactions on Intelligent Transportation Systems (T-ITS), **[Accepted paper, to be published in Dec 2017]**. [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)


## Packages

* [train](train) contains tools for training the network for semantic segmentation.
* [eval](eval) contains tools for evaluating/visualizing the network's output.
* [trained_models](trained_models) Contains the trained models used in the papers.

## Requirements
The Cityscapes dataset, which can be downloaded [here](https://www.cityscapes-dataset.com/).

Torch Library [(installation tutorial)](http://torch.ch/docs/getting-started.html) with CUDA and CuDNN backends.

NOTE: The code has been tested in Ubuntu 16.04 and Torch7 with CUDA 8.0 and CuDNN 5.1. It should work with other versions but we cannot guarantee it.

## Usage

Usage and examples for either [training](train) or [evaluating](eval) the models are described in the READMEs of each section.


## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/

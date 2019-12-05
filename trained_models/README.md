# Trained ERFNet models

These are the trained models that were used in the papers:

* [erfnet_pretrained.net](erfnet_pretrained.net) ERFNet full model with encoder trained on ImageNet and decoder trained on Cityscapes train set.
* [erfnet_scratch.net](erfnet_scratch.net) ERFNet full model with encoder trained from scratch on Cityscapes train set (1/8 labels) and decoder trained on Cityscapes train set.
* [pretrained_encoder_imagenet.net](pretrained_encoder_imagenet.net)	ERFNet encoder trained on ImageNet and used to train '_pretrained' model. Can be used to train ERFNet decoder with the training code by specifying this file in the '--CNNEncoder' option.


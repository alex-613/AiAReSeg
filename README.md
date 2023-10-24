# AiAReSeg: Catheter tracking and segmentation in intraoperative ultrasound using transformers

[Alex Ranne](https://github.com/alex-613), [Yordanka Velikova](https://github.com/danivelikova), Nassir Navab, Ferdinando Rodriguez y Baena

[ArXiv Preprint] (https://arxiv.org/abs/2309.14492)

## Highlight

![](AiAReSeg.png)

AiAReSeg is a novel deep learning semantic segmenttaion architecture based on ResNet, 3D UNet, transformers, and the Attention in Attention (AiA) module.
While most segmentation task involves labelling a single frame, without any temporal context, at a time through the network, AiAReSeg instead is optimised at
processing a sequence of frames at the same time, and drawing on information from past frames in order to make its prediction on the current frame.
While we have designed the architecture specifically for the segmentation of interventional medical ultrasound, this framework is applicable for other sequential data
and videos of moving objects.

## Brief overview of AiAReSeg architecture

The network consists of three input branches: Initial frame, search frame, and intermediate frames. These correspond to the
first, current and 2 chosen intermediate frames between the two.

The main architecture consists of 4 components: ResNet, Transformer, the Segmentation Head and the skip connections.

**ResNet:** ResNet acts as the default feature extractor that converts the input images into feature maps. 

**Transformer:** The transformer selected in our architecture is the [AiATrack](https://github.com/Little-Podi/AiATrack) network. In short, AiATrack consists of three
branches, similiar to AiAReSeg, and consists of a standard self attention encoder, with a two module cross attention decoder responsible for processing the
short (cross attention between the features of the current frame and the intermediate frames) and long (cross attention between the features of the current frame and the initial
frame) term attention. The output of the transformer is a flattened weight array.

**Segmentation Head:** The segmentation head was bespokely designed to combine temporal features into a single feature channel. From the output
of the transformer, the weight array is reshaped into an image patch, then stacked with feature maps from the ResNet encoder at different scales.
Unlike the UNet which concatenates dissimiliar features in the same channel, the segmentation head stacks them along a new time dimension, then performs 3D deconvolution to 
condense the time dimension, before stacking again at the next scale level. For example, with an intermediate frame number of 2, the time dimension will have a size of 5 
(1 current frame, 1 initial frame, 2 intermediate frames, and 1 frame the transformer output).

**Skip connections:** The skip connections have the important job of matching the sizes of the non-time dimension via a series of 2D convolutional layers.

## Getting started:

Please follow the instructions detailed here to run the model on your local PC. This model was tesetd on Ubuntu 20.04 with CUDA 11.3.

## Installation:

    git clone https://github.com/alex-613/AiAReSeg
    cd AiAReSeg
    pip install -r environment.yml
    sudo apt-get install ninja-build
    sudo apt-get install libturbojpeg

Note that Pytorch version must be <= 1.10.1.

## Data Preparation:

The data used in this work are ray casting simulations based on open source CT data, obtained from the Synapse online database.
These data will be made available here in the near future, please stay tuned!

Please ensure that your data is in the following format:

    --ROOT/
        |--Catheter
            |--Catheter-1
            |--Catheter-2
            ...

Edit the **PATH** in ```lib/test/evaluation/local.py``` and ```lib/train/adim/local.py``` to the proper absolute path.

## Training:
Please run the following command from ROOT:

    python lib/train/run_training.py --segmentation 1 --config AiASeg_s+p --script aiareseg

Training results will be saved under lib/train/checkpoints
## Testing:

Before running testing, please modify the paths of the yaml configuration file and the model checkpoint to where they are located on your PC.
Edit these paths in ```lib/test/parameter/aiareseg.py```.

Once that is modified run the following command:

    python tracking/test.py --tracker aiareseg --dataset catheter_segmentation_test --param AiASeg_s+p --segmentation 1

## Acknowledgements:

Our work is implemented based on the following projects, please refer to their works for further details:

- [AiATrack](https://github.com/Little-Podi/AiATrack)
- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- [UNet](https://arxiv.org/abs/1505.04597)
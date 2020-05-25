Implementation of "CHANNEL-WISE PROGRESSIVE LEARNING FOR LOSSLESS IMAGE COMPRESSION"

Hochang Rhee, Yeong Il Jang, Seyun Kim, and Nam Ik Cho

## Environments
- Ubuntu 18.04
- [Tensorflow 1.9](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.6.5
- Python 3.5.5

## Abstract

This paper presents a channel-wise progressive coding system for lossless compression of color images. We follow the classical lossless compression scheme of LOCO-I and CALIC, where pixel values and coding contexts are predicted and forwarded to the entropy coder for compression. The contribution is that we jointly estimate the pixel values and coding contexts from neighboring pixels by training a simple multilayer perceptron in a residual and channel-wise progressive manner. Specifically, we obtain accurate pixel prediction along with coding contexts that reflect the magnitude of local activity very well. These results are sent to an adaptive arithmetic coder that appropriately encodes the prediction error according to the corresponding coding context. Experimental results demonstrate the effectiveness of the proposed method in high-resolution datasets.
<br><br>

## Proposed Method

### <u>Overall framework of proposed method</u>

<p align="center"><img src="figure/Overall.png" width="700"></p>

During meta-transfer learning, the external dataset is used, where internal learning is done during meta-test time.
From random initial \theta_0, large-scale dataset DIV2K with “bicubic” degradation is exploited to obtain \theta_T.
Then, meta-transfer learning learns a good representation \theta_M for super-resolution tasks with diverse blur kernel scenarios.
In the meta-test phase, self-supervision within a test image is exploited to train the model with corresponding blur kernel.

### <u> Algorithms </u>

<p align="center"><img src="figure/meta-training.png" width="400">&nbsp;&nbsp;<img src="figure/meta-test.png" width="400"></p> 

Left: The algorithm of Meta-Transfer Learning & Right: The algorithm of Meta-Test.

## Experimental Results

**Results on various kernel environments (X2)**

<p align="center"><img src="figure/result.png" width="900"></p>

The results are evaluated with the average PSNR (dB) and SSIM on Y channel of YCbCr colorspace.
<font color="red">Red </font> color denotes the best results and <font color ="blue"> blue </font> denotes the second best.
The number between parantheses of our methods (MZSR) denote the number of gradient updates.

**Results on scaling factor (X4)**

<p align="center"><img src="figure/resultx4.png" width="900"></p>

**Test Input Data**

Degraded Images of Set5, B100, Urban100 on various kernel environments.

[Download](https://drive.google.com/open?id=16L961dGynkraoawKE2XyiCh4pdRS-e4Y)

## Visualized Results

<p align="center"><img src="figure/001.png" width="900"></p>
<br><br>
<p align="center"><img src="figure/002.png" width="900"></p>

## Brief explanation of contents

```
├── GT: Ground-truth images
├── Input: Input LR images
├── Model: Pre-trained models are included (Model Zoo)
    ├──> Directx2: Model for direct subsampling (x2)
    ├──> Multi-scale: Multi-scale model
    ├──> Bicubicx2: Model for bicubic subsampling (x2)
    └──> Directx4: Model for direct subsampling (x4)
├── Pretrained: Pre-trained model (bicubic) for transfer learning.
└── results: Output results are going to be saved here.

Rest codes are for the training and test of MZSR.
```

## Guidelines for Codes

**Requisites should be installed beforehand.**

Clone this repo.
```
git clone http://github.com/JWSoh/MZSR.git
cd MZSR/
```

### Training

Download training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

#### Generate TFRecord dataset
- Refer to [MainSR](https://www.github.com/JWSoh/MainSR) repo.
- Run generate_TFRecord_MZSR.py

#### Train MZSR
Make sure all configurations in **config.py** are set.

[Options]
```
python main.py --train --gpu [GPU_number] --trial [Trial of your training] --step [Global step]

--train: Flag in order to train.
--gpu: If you have more than one gpu in your computer, the number denotes the index. [Default 0]
--trial: Trial number. Any integer numbers can be used. [Default 0]
--step: Global step. When you resume the training, you need to specify the right global step. [Default 0]
```

### Test

Ready for the input data (low-resolution) and corresponding kernel (kernel.mat file.)

[Options]
```
python main.py --gpu [GPU_number] --inputpath [LR path] --gtpath [HR path] --savepath [SR path]  --kernelpath [kernel.mat path] --model [0/1/2/3] --num [1/10]

--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--inputpath: Path of input images [Default: Input/g20/Set5/]
--gtpath: Path of reference images. [Default: GT/Set5/]
--savepath: Path for the output images. [Default: results/Set5]
--kernelpath: Path of the kernel.mat file. [Default: Input/g20/kernel.mat]
--model: [0/1/2/3]
    -> 0: Direct x2
    -> 1: Multi-scale
    -> 2: Bicubic x2
    -> 3: Direct x4
--num: [1/10] The number of adaptation (gradient updates). [Default 1]

```

You may change other minor options in "test.py."
Line 9 to line 17.

The minor options are shown below.
```
self.save_results=True		-> Whether to save results or not.
self.display_iter = 1		-> The interval of information display.
self.noise_level = 0.0		-> You may sometimes add small noise for real-world images.
self.back_projection=False	-> You may also apply back projection algorithm for better results.
self.back_projection_iters=4	-> The number of iteration of back projection.
```

### An example of test codes

```
python main.py --gpu 0 --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat --model 0 --num 1
```

## Citation
```
@article{soh2020meta,
  title={Meta-Transfer Learning for Zero-Shot Super-Resolution},
  author={Soh, Jae Woong and Cho, Sunwoo and Cho, Nam Ik},
  journal={arXiv preprint arXiv:2002.12213},
  year={2020}
}

@inproceedings{soh2020meta,
  title={Meta-Transfer Learning for Zero-Shot Super-Resolution},
  author={Soh, Jae Woong and Cho, Sunwoo and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgement
Our work and implementations are inspired by and based on
ZSSR [[site](https://github.com/assafshocher/ZSSR)] and MAML [[site](https://github.com/cbfinn/maml)].

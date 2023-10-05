# Pag-Unet
##  :scroll: Introduction
This directory contains codes of our paper Pag-Unet:
- Pag-Unet uses an efficient multidimensional feature aggregation method to solve the problem of using naive operations to aggregate features in current multitasking transform models, which cannot efficiently utilize the features obtained by encoders.
- Pag-Unet proposed a new personalized self attention method.
- Pag-Unet achieves superior performance on NYUD-v2 and PASCAL-Context datasets respectively, and significantly outperforms previous state-of-the-arts.

<p align="center">
  <img alt="img-name" src="https://github.com/UPLI-123/Pag-Unet/blob/main/Image/Pag-Unet.png" width="700">
  <br>
    <em>Framework overview of the proposed Pixel-attention-guided Unet (Pag-Unet) for dense scene understanding.</em>
</p>

<p align="center">
  <img alt="img-name" src="https://github.com/UPLI-123/Pag-Unet/blob/main/Image/Pself.png" width="1000">
  <br>
    <em>Framework overview of the proposed Personalized self-attention.</em>
</p>

# :grinning: Configuration!
## Get Data
We use the same data (PASCAL-Context and NYUD-v2) as [InvPT](https://github.com/prismformore/Multi-Task-Transformer/blob/main/InvPT). You can download the data from:
[PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab),
[NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c)
### **Special evaluation for boundary detection**
Use the evaluation tools in this [repo](https://github.com/prismformore/Boundary-Detection-Evaluation-Tools).

We follow previous works and use Matlab-based [SEISM](https://github.com/jponttuset/seism) project to compute the optimal dataset F-measure scores. The evaluation code will save the boundary detection predictions on the disk. 

Specifically, identical to ATRC and ASTMT, we use [maxDist](https://github.com/jponttuset/seism/blob/6af0cad37d40f5b4cbd6ca1d3606ec13b176c351/src/scripts/eval_method.m#L34)=0.0075 for PASCAL-Context and maxDist=0.011 for NYUD-v2. Thresholds for HED (under seism/parameters/HED.txt) are used. ```read_one_cont_png``` is used as IO function in SEISM.
# :partying_face:	 Pre-trained Pag-Unet models
To faciliate the community to reproduce our SoTA results, we re-train our best performing models with the training code in this repository and provide the weights for the reserachers.

### Download pre-trained models
|Version | Dataset | Download | Segmentation | Human parsing | Saliency | Normals | Boundary | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Pag-Unet(our paper)**| **PASCAL-Context** | [Baidu Netdisk](https://pan.baidu.com/s/1Z3knCJ4EYHzMhp4VPD7tTA?pwd=9yih) | **79.12** | **70.24** | 83.88 | **14.04** | **75.79** |
| InvPT (ECCV 2022)| PASCAL-Context | - | 79.03 | 67.61 | **84.81** | 14.15 | 73.00 | 
| DeMT (AAAI 2023) | PASCAL-Context | - | 75.33 | 63.11 | 83.42 | 14.54 | 73.20 |

|Version | Dataset | Download | Segmentation | Depth | Normals | Boundary|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Pag-Unet(our paper)**| **NYUD-v2** | [Baidu Netdisk](https://pan.baidu.com/s/1Wzen31ivqMTV6kWPJ_XCmA?pwd=362c) | **53.94** | 0.5456 | **18.90** | **78.65**|
|InvPT (ECCV 2022) |NYUD-v2|-| 53.56 | **0.5183** | 19.04 | 78.10 |
|DeMT (AAAI 2023) |NYUD-v2|-| 51.50 | 0.5474 | 20.02 | 78.10|

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
| **InvPT<sup>*</sup>**| **PASCAL-Context** | [google drive](https://drive.google.com/file/d/1r0ugzCd45YiuBrbYTb94XVIRj6VUsBAS/view?usp=sharing), [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EcwMp9uUEfdLnQcaNJsN3bgBfQeHHqs2pkj7KmtGx_dslw?e=0CtDfq) | **79.91** | **68.54** | **84.38** | **13.90** | **72.90** |
| InvPT (our paper) | PASCAL-Context | - | 79.03 | 67.61 | 84.81 | 14.15 | 73.00 | 
| ATRC (ICCV 2021) | PASCAL-Context | - | 67.67 | 62.93 | 82.29 | 14.24 | 72.42 |

|Version | Dataset | Download | Segmentation | Depth | Normals | Boundary|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **InvPT<sup>*</sup>**| **NYUD-v2** | [google drive](https://drive.google.com/file/d/1Ag_4axN-TaAZS_W-nFIm4__DoDw1zgqI/view?usp=sharing), [onedrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EU6ypDGEFPFLuC5rG5Vj2KkBliG1gXgbXh2t_YQJIk9YLw?e=U6hJ4H) | **53.65** | **0.5083** | **18.68** | **77.80**|
|InvPT (our paper) |NYUD-v2|-| 53.56 | 0.5183 | 19.04 | 78.10 |
| ATRC (ICCV 2021) |NYUD-v2|-| 46.33 | 0.5363 | 20.18 | 77.94|

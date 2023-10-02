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
We use the same data (PASCAL-Context and NYUD-v2) as ATRC. You can download the data from:
[PASCALContext.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/ER57KyZdEdxPtgMCai7ioV0BXCmAhYzwFftCwkTiMmuM7w?e=2Ex4ab),
[NYUDv2.tar.gz](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hyeae_connect_ust_hk/EZ-2tWIDYSFKk7SCcHRimskBhgecungms4WFa_L-255GrQ?e=6jAt4c)


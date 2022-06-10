

<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Spectral Propagation Network - A Novel Dual Information Propagation System for Hyperspectral Image Classification</h3>

  <p align="center">
    A Novel Spctral-Spatial Attention Mechanism for Hyperspectral Image Classification!
    <br>
    <br>
    <a href=[https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/github/Naereen/badges)><img src="https://colab.research.google.com/assets/colab-badge.svg" alt =" Open in Colab" height = "20"></a>
    ·
    <a href=[https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg](https://creativecommons.org/licenses/by-nc-nd/4.0/)><img src="https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg" alt = "License: CC BY-NC-ND 4.0" height = "20"></a>
    .
    <a href=[https://img.shields.io/github/forks/Naereen/StrapDown.js.svg?style=social&label=Fork&maxAge=2592000](https://GitHub.com/Naereen/StrapDown.js/network/)><img src="https://img.shields.io/github/forks/Naereen/StrapDown.js.svg?style=social&label=Fork&maxAge=2592000" height = "20"></a>
    .
      <a href=[https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Fork&maxAge=2592000](https://GitHub.com/Naereen/StrapDown.js/network/)><img src="https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Star&maxAge=2592000" height = "20"></a>
  </p>
</div>

Description : This research project introduce a novel attention mechanism that focuses on different receptive regions for both spectral and spatial domain. Empirical analysis on three benchmark datasets prove the superiority of the proposed spectral propagation network for hyperspectral image classification. The contribution of the featured spectral propagation network is given below :
- A novel attention mechanism has been introduced leveraging dynamic kernels in order to harvest various receptive regions.
- The proposed attention module is a plug-in module that can learn representations from both spectral and spatial domains.
- The spectral propagation module can harvest global information and learns only useful spectral-spatial features preserving long-term information.

## Training
For spectral propagation network :

```
python main.py --dataset IP --tr_percent 0.1
```




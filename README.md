# Description
Implementation of the Latent ODE (https://github.com/YuliaRubanova/latent_ode) for continuous glucose monitor (CGM) data.

# Requirements

Install required packages by running `conda env create -f environment.yml`. Additionally, install `torchdiffeq` from [here](https://github.com/rtqichen/torchdiffeq).

# Interpolating CGM Data

<p align="center">
<img align="middle" src="./plots/20230426-204841.gif" width="800" />
</p>

# Extrapolating CGM Data

<p align="center">
<img align="middle" src="./plots/20230426-204954.gif" width="800" />
</p>

# Data

The data set has been taken from the study by [1] and can be downloaded from the [here](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005143#pbio.2005143.s010).

`[1] Heather Hall, Dalia Perelman, Alessandra Breschi, Patricia Limcaoco, Ryan Kellogg, Tracey McLaughlin, and Michael Snyder. Glucotypes reveal new patterns of glucose dysregulation. *PLoS biology*, 16(7):e2005143, 2018.`
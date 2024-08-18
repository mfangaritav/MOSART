![mosart_icon](https://github.com/user-attachments/assets/8512c53c-c44a-4c91-9c96-89885f806100)

# Morphology from SAR toolbox
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6784982.svg)](https://doi.org/10.5281/zenodo.6784982)
<a target="_blank" href="https://colab.research.google.com/github/mfangaritav/MOSART/blob/main/Process_ALOS.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Authors: Mario Angarita, Ronni Grapenthin, Franz Meyer, Simon Plank and Michael Christoffersen

## What is MoSARt?

The Morphology from SAR toolbox (MoSARt) is a set of python scripts to retrieve morphology changes from SAR amplitude images.

![workflow_mosart](https://github.com/user-attachments/assets/4e99432b-1770-4896-9ffc-31def03f0aed)

The user needs to provide a set of coregistered amplitude images and a Digital Elevation Model in radar coordinates,
the user can also provide longitude and latitude datasets to georeference the results.

## Standard Installation (without coregistration workflow)

We recommend using Anaconda or Miniconda to create an environment in which to install MoSARt to prevent dependency conflicts.

```console
conda create --n mosart
pip install git+https://github.com/mfangaritav/MOSART.git
```

## Full Installation

MoSARt provides functions to coregister images from ALOS-1 and Sentinel-1 images (bursts). To run these functions the user needs
to install [hyp3-isce2](https://github.com/ASFHyP3/hyp3-isce2).

```console
git clone https://github.com/ASFHyP3/hyp3-isce2.git
cd hyp3-isce2
mamba env create -f environment.yml
mamba activate hyp3-isce2
python -m pip install -e .
```

## Obtain elevation changes:

This repository contains a [notebook](Process_ALOS.ipynb) to illustrate the full workflow from MoSARt to retrieve elevation changes for the 2008 Okmok caldera eruption.

# Publication:

MoSARt was used to retrieve the elevation changes in the 2019-2020 Shishaldin eruption:

- Angarita, M., Grapenthin, R., Plank, S., Meyer, F. J., & Dietterich, H. (2022). Quantifying large‐scale surface change using SAR amplitude images: Crater morphology changes during the 2019–2020 Shishaldin volcano eruption. Journal of Geophysical Research: Solid Earth, 127(8), e2022JB024344. https://doi.org/10.1029/2022JB024344

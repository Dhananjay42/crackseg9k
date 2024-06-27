# CrackSeg9k: A Collection and Benchmark for Crack Segmentation Datasets and Frameworks
[![Paper](https://img.shields.io/badge/Papers%20With%20Code-21CBCE.svg?style=for-the-badge&logo=Papers-With-Code&logoColor=white)](https://paperswithcode.com/paper/crackseg9k-a-collection-and-benchmark-for/review/)
[![Paper](https://img.shields.io/badge/Paper-CrackSeg9k-green)](https://doi.org/10.1007/978-3-031-25082-8_12)
[![Dataset](https://img.shields.io/badge/Dataset-Harvard%20Dataverse-blue)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY)


Authors: [Shreyas Kulkarni](https://github.com/shreyask3107), [Shreyas Singh](https://github.com/shreyesss), [Dhananjay Balakrishnan](https://github.com/Dhananjay42), Siddharth Sharma, [Saipraneeth Devunuri](https://github.com/praneethd7), Sai Chowdeswara Rao Korlapati.

### About
This repository consists of codes to replicate the analysis in [CrackSeg9k](https://doi.org/10.1007/978-3-031-25082-8_12) paper presented at ECCV 2022 conference. The codes for training and evaluation for each image segmentation model are present in their respective folders. The dataset of ~9160 images (hence the name CrackSeg9k) is publicly available on [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY). Due to size restrictions on harvard dataverse, the whole dataset of ~9k images is split in two sub-folders. Make sure to download the `version V4` (as of June 2024) and extract both sub folders for the whole dataset. If you use the paper, code or dataset in your research, please consider citing using the appropriate Bibtex citations provided in the [Citation](#citation) section below.

#### Abstract
> The detection of cracks is a crucial task in monitoring structural health and ensuring structural safety. The manual process of crack detection is time-consuming and subjective to the inspectors. Several researchers have tried tackling this problem using traditional Image Processing or learning-based techniques. However, their scope of work is limited to detecting cracks on a single type of surface (walls, pavements, glass, etc.). The metrics used to evaluate these methods are also varied across the literature, making it challenging to compare techniques. This paper addresses these problems by combining previously available datasets and unifying the annotations by tackling the inherent problems within each dataset, such as noise and distortions. We also present a pipeline that combines Image Processing and Deep Learning models. Finally, we benchmark the results of proposed models on these metrics on our new dataset and compare them with state-of-the-art models in the literature.
## Instructions:

1. Code for each model presented in the paper is available in the individual folders.
2. Feature generation with DINO is available in the "dino" folder.


## Citation

### Paper
```
@inproceedings{kulkarni2022crackseg9k,
  title={CrackSeg9k: a collection and benchmark for crack segmentation datasets and frameworks},
  author={Kulkarni, Shreyas and Singh, Shreyas and Balakrishnan, Dhananjay and Sharma, Siddharth and Devunuri, Saipraneeth and Korlapati, Sai Chowdeswara Rao},
  booktitle={European Conference on Computer Vision},
  pages={179--195},
  year={2022},
  organization={Springer}
}
```
### Dataset
```
@data{DVN/EGIEBY_2022,
author = {Siddharth Sharma and Dhananjay Balakrishnan and Shreyas Kulkarni and Shreyas Singh and Saipraneeth Devunuri and Sai Chowdeswara Rao Korlapati},
publisher = {Harvard Dataverse},
title = {{Crackseg9k: A Collection of Crack Segmentation Datasets}},
year = {2022},
version = {V4},
doi = {10.7910/DVN/EGIEBY},
url = {https://doi.org/10.7910/DVN/EGIEBY}
}
```

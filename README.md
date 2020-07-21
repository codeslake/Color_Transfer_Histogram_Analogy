## Deep Color Transfer using Histogram Analogy &mdash; Official PyTorch Implementation
![Python 2.7.12](https://img.shields.io/badge/python-2.7.12-green.svg?style=plastic)
![PyTorch 0.4.0](https://img.shields.io/badge/PyTorch-0.4.0-green.svg?style=plastic)
![CUDA 8.0.61](https://img.shields.io/badge/CUDA-8.0.61-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

![Teaser image](./assets/figure.jpg)
**Picture:** *Color transfer results on various source and reference image pairs. For visualization, the reference image is cropped to make a same size with other images.*

This repository contains the official PyTorch implementation of the following paper:

> **Deep Color Transfer using Historgram Analogy**<br>
> Junyong Lee (POSTECH), Hyeongseok Son (POSTECH), Gunhee Lee (NCSoft), Jonghyeop Lee (POSTECH), Sunghyun Cho (POSTECH) Seungyong Lee (POSTECH)
> 
> http://cg.postech.ac.kr/papers/2020_CGI_JY.pdf
>
> **Abstract:** *We propose a novel approach to transferring the color of a reference image to a given source image. Although there can be diverse pairs of source and reference images in terms of content and composition similarity,previous methods are not capable of covering the whole diversity. To resolve this limitation, we propose a deep neural network that leverages color histogram analogy for color transfer. A histogram contains essential color information of an image, and our network utilizes the analogy between the source and reference histograms to modulate the color of the source image with abstract color features of the reference image. In our approach, histogram analogy is exploited basically among the whole images, but it can also be applied to semantically corresponding regions in the case that the source and reference images have similar contents with different compositions. Experimental results show that our approach effectively transfers the reference colors to the source images in a variety of settings. We also demonstrate a few applications of our approach, such asalette-based recolorization, color enhancement, and color editing.*

For any inquiries, please contact [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://drive.google.com/file/d/1mRVo3JefkgRd2VdJvG5M-8xWtvl60ZWg/view?usp=sharing) |
| [Supplementary Files](https://drive.google.com/file/d/1sQTGHEcko2HxoIvneyrot3bUabPrN5l1/view?usp=sharing) |
| [Checkpoint Files](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing) |


## Testing the network
To test the network, type
```bash
python test.py --dataroot [test folder path] --checkpoints_dir [ckpt path]
```

## Using pre-trained networks
Download pretrained weights for IRN and HEN from [here](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing).
Place the file under `./checkpoints` (one may change the offset in `./options/base_options.py`).

## BIBTEX
If you find this code useful, please consider citing:

```
@InProceedings{Lee_2020_VC,
author = {Lee, Junyong and Son, Hyeongseok and Lee, Gunhee and Lee, Jonghyeop and Cho, Sunghyun and Lee, Seungyong},
title = {Deep Color Transfer using Histogram Analogy},
booktitle = {The Visual Computer},
month = {Aug},
year = {2020}
}
```

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition,ersonalization technology through userreference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

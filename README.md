# Deep Color Transfer using Histogram Analogy
![Python 2.7.12](https://img.shields.io/badge/python-2.7.12-green.svg?style=plastic)
![PyTorch 0.4.0](https://img.shields.io/badge/PyTorch-0.4.0-green.svg?style=plastic)
![CUDA 8.0.61](https://img.shields.io/badge/CUDA-8.0.61-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

![Teaser image](./assets/figure.jpg)
**Figure:** *Color transfer results on various source and reference image pairs. For visualization, the reference image is cropped to make a same size with other images.*

This repository contains the official PyTorch implementation of the following paper:

> **[Deep Color Transfer using Histogram Analogy](http://cg.postech.ac.kr/papers/2020_CGI_JY.pdf)**<br>
> Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho and Seungyong Lee, CGI2020

If you find this code useful, please consider citing:
```
@article{Lee_2020_CTHA,
  author = {Lee, Junyong and Son, Hyeongseok and Lee, Gunhee and Lee, Jonghyeop and Cho, Sunghyun and Lee, Seungyong},
  title = {Deep Color Transfer using Histogram Analogy},
  journal = {The Visual Computer},
  volume = {36},
  number = {10},
  pages = {2129--2143},
  year = 2020,
}
```


For any inquiries, please contact [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://drive.google.com/file/d/1mRVo3JefkgRd2VdJvG5M-8xWtvl60ZWg/view?usp=sharing) |
| [Supplementary Files](https://drive.google.com/file/d/1sQTGHEcko2HxoIvneyrot3bUabPrN5l1/view?usp=sharing) |
| [Checkpoint Files](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing) |


## Testing the network
1. Download pretrained weights for IRN and HEN from [here](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing).
Then, place checkpoints under `./checkpoints` (one may change the offset in `./options/base_options.py`).

2. Place your images under `./test`. Input images and their segment map should be placed under `./test/input` and `./test/seg_in`, respectively. Place target images and their segment map under `./test/target` and `./test/seg_tar`. 

3. To test the network, type
```bash
python test.py --dataroot [test folder path] --checkpoints_dir [ckpt path]
# e.g., python test.py --dataroot test --checkpoints_dir checkpoints
```
4. To turn of *sementaic replacement*, add `--is_SR`.
```bash
python test.py --dataroot [test folder path] --checkpoints_dir [ckpt path] --is_SR
```


## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

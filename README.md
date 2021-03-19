# Deep Color Transfer using Histogram Analogy
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

![Teaser image](./assets/figure.jpg)
**Figure:** *Color transfer results on various source and reference image pairs. For visualization, the reference image is cropped to make a same size with other images.*

This repository contains the official PyTorch implementation of the following paper:

> **[Deep Color Transfer using Histogram Analogy](https://link.springer.com/epdf/10.1007/s00371-020-01921-6?sharing_token=m2UzXwVlSCP8CbRYNrEcnve4RwlQNchNByi7wbcMAY5_mQV2iPdNT8_ORizvbX3p8mina4UHEjoKsvegf0S_FwC3Yo3cBRV6mlx1mdbvv3CiiREpz3ZqyJuRGmHbygkNL_7X-3hd2CMGSxgPtF22LPsyjpEfhG1R_bNHSSVNvbc%3D)**<br>
> Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho and Seungyong Lee, CGI2020


## Getting Started
### Prerequisites
*Tested environment*

![Ubuntu](https://img.shields.io/badge/Ubuntu-18.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8.0-green.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2-green.svg?style=plastic)

1. **Install requirements**
    * `pip install -r requirements.txt`

2. **Pre-trained models**
    * Download and unzip [pretrained weights](https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=1) under `[CHECKPOINT_ROOT]`:

        ```
        ├── [CHECKPOINT_ROOT]
        │   ├── *.pth
        ```

        > **NOTE:**
        > 
        > `[CHECKPOINT_ROOT]` can be specified with the option `--checkpoints_dir`.


## Testing the network
1. Place input images and their segment maps should be placed under `./test/input` and `./test/seg_in`, respectively. Place target images and their segment maps under `./test/target` and `./test/seg_tar`, respectively. 

2. Test the network:

```bash
python test.py --dataroot [test folder path] --checkpoints_dir [CHECKPOINT_ROOT]
# e.g., python test.py --dataroot test --checkpoints_dir checkpoints
```
    * The test results will be saved under `./results/`.
    * To turn on *semantic replacement*, add `--is_SR`.

    ```bash
    python test.py --dataroot [test folder path] --checkpoints_dir [ckpt path] --is_SR
    ```

## Citation
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

## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://link.springer.com/epdf/10.1007/s00371-020-01921-6?sharing_token=m2UzXwVlSCP8CbRYNrEcnve4RwlQNchNByi7wbcMAY5_mQV2iPdNT8_ORizvbX3p8mina4UHEjoKsvegf0S_FwC3Yo3cBRV6mlx1mdbvv3CiiREpz3ZqyJuRGmHbygkNL_7X-3hd2CMGSxgPtF22LPsyjpEfhG1R_bNHSSVNvbc%3D) |
| [Supplementary Files](http://cg.postech.ac.kr/papers/2020_CGI_JY_supp.pdf) |
| [Checkpoint Files](https://www.dropbox.com/s/lkwo9xg168e650i/checkpoints.zip?dl=1) |

<!-- https://www.dropbox.com/s/8ty3lfqa27e5b5l/202010_Deep%20Color%20Transfer%20using%20Histogram%20Analogy.pdf?raw=1 -->

## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)

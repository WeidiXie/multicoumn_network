# Multicolumn Networks

This repo contains a Keras code of the paper,

[Multicolumn Networks for Face Recognition (Xie and Zisserman, BMVC2018)](https://arxiv.org/pdf/1807.09192.pdf).


### Dependencies
- [python]
- [Keras 2.2.4](https://keras.io/)
- [Tensorflow 1.8.0](https://www.tensorflow.org/)


### Meta & Model

Keras Model (https://www.dropbox.com/sh/glh67gh049m86c2/AAB_PYuwPiCv557juxIAZqYLa?dl=0),

IJBB Dataset (preprocessed) (https://www.dropbox.com/s/t42qmx1yef00iia/ijbb_crop.tar.gz?dl=0)

Meta Information (https://www.dropbox.com/sh/3tk5fjl6i08u9xi/AAD5Dzb340RYW9eV4iY1Vdnea?dl=0)


### Results on IJBB verification (Table 3 in the paper):

| Arch   | Feat dim | Pretrain | TAR@FAR = 1e-5 | TAR@FAR = 1e-4 | TAR@FAR = 1e-3 | TAR@FAR = 1e-2 | TAR@FAR = 1e-1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|   MN-v    | 512 |  N  | 0.683 | 0.818 | 0.902 | 0.955 | 0.984 |
|   MN-vc    | 512 |  N  | 0.708 | 0.831 | 0.909 | 0.958 | 0.985 |


### Testing the model
To test the model on the IJBB dataset, 

- python predict.py 

### Citation
```
@InProceedings{Xie18,
  author       = "W. Xie, A. Zisserman ",
  title        = "Multicolumn Networks for Face Recognition",
  booktitle    = "British Machine Vision Conference (BMVC), 2018.",
  year         = "2018",
}
```


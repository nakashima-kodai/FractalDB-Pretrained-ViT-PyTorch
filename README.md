# FractalDB Pretrained ViT
This repo is the official implementation of ["Can Vision Transformers Learn without Natural Images?"](https://arxiv.org/abs/2103.13023). This repo is based on the [timm](https://github.com/rwightman/pytorch-image-models) and [DeiT](https://github.com/facebookresearch/deit).

We clarify that FractalDB pre-trained ViT can achieve a competitive validation accuracy with ImageNet pre-trained ViT. FractalDB consist of automatically generated image patterns and their labels based on a mathematical formula.
![acc_transition](figures/acc_transition.png)


# Model Zoo
Comming soon...

# Usage
## Pre-training

```
python pretrain.py data={dataset name for pre-training} data.set.root={/path/to/dataset}
```

## Fine-tuning


# Citation
```
@inproceedings{Nakashima_arXiv2021,
 author = {Nakashima, Kodai and Kataoka, Hirokatsu and Matsumoto, Asato and Iwata, Kenji and Inoue, Nakamasa},
 title = {Can Vision Transformers Learn without Natural Images?},
 booktitle = {CoRR:2103.13023},
 year = {2021}
}
```
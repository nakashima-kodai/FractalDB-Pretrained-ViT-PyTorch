# FractalDB Pretrained ViT
This repo is the official implementation of ["Can Vision Transformers Learn without Natural Images?"](https://arxiv.org/abs/2103.13023). This repo is based on the [timm](https://github.com/rwightman/pytorch-image-models) and [DeiT](https://github.com/facebookresearch/deit).

We clarify that FractalDB pre-trained ViT can achieve a competitive validation accuracy with ImageNet pre-trained ViT. FractalDB consist of automatically generated image patterns and their labels based on a mathematical formula.
![acc_transition](figures/acc_transition.png)



# Usage
## Pre-training
Run the code ```pretrain.py``` to create a FractalDB pre-trained model.
```
python pretrain.py data={dataset name for pre-training} data.set.root={/path/to/dataset}
```
The pre-trained models will be shared within the next week.


## Fine-tuning
Run the code ```finetune.py``` to additionally train any image datasets.
```
python finetune.py data={dataset name for fine-funing} data.set.root={/path/to/dataset}
```

# Citation
```
@inproceedings{Nakashima_arXiv2021,
 author = {Nakashima, Kodai and Kataoka, Hirokatsu and Matsumoto, Asato and Iwata, Kenji and Inoue, Nakamasa},
 title = {Can Vision Transformers Learn without Natural Images?},
 booktitle = {CoRR:2103.13023},
 year = {2021}
}
```
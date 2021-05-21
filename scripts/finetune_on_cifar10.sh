DATA=cifar10
DATAROOT={/path/to/cifar10}
CKTP=outputs/{pretrain_date}/{filename}.pth


python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10

# Training Regular Polytope Networks with knowledge distillation
Training a RePoNet from scratch can easily become an unfeasible and costly task since there are no available public pretrained models with that kind of architecture.<br />
We address this issue by transferring the knowledge from a standard Neural Network with great performances on a given dataset to the RePoNet following classic knowledge distillation and its guidelines. The experiments performed by the following code explore the use of this technique using different architectures and different datasets while testing the previously mentioned guidelines.<br />
More detail in the paper: [Chisci Marco Training Regular Polytope Networks with Knowledge Distillation.pdf](https://github.com/marcochisciedu/Training-Regular-Polytope-Networks-with-knowledge-distillation/files/14852038/Chisci.Marco.Training.Regular.Polytope.Networks.with.Knowledge.Distillation.pdf)


### Installing

To run this project you need Python>=3.6 installed on your machine.<br />

Install python dependencies by running:


```
pip install -r requirements.txt
```

## Experiments

### Fine-tuning

To fine-tune the BiT models I used the code in the official repository of [Big Transfer (BiT): General Visual Representation Learning](https://github.com/google-research/big_transfer?tab=readme-ov-file). (bit_pytorch, bit_common.py and bit_hyperrule.py)

Run the following line to dowload the ResNet pretrained model:

```
wget https://storage.googleapis.com/bit_models/BiT-M-R152x2.npz
```

To fine-tune the teacher model (e.g. fine-tuning on cifar10):

```
python3 -m bit_pytorch.train --name cifar10_run --model BiT-M-R152x2 --logdir bit_logs --dataset cifar10 --bit_pretrained_dir BiT-M-R152x2.npz --datadir datadir 
```

To fine-tune the ViT models I used the code in [ViT Pytorch](https://github.com/jeonsworld/ViT-pytorch) (vit_models, vit_utils, vit_train.py).

The following line dowloads the ViT pretrained model:

```
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
```

To fine-tune the teacher model(e.g. fine tuning on Oxford flowers102):

```
python3 vit_train.py --name flowers102_run --dataset oxford_flowers102 --datadir datadir --model_type ViT-B_32 --pretrained_dir ViT-B_32.npz 
```


### Knowledge distillation

The distillation code was written using [FunMatch Distillation](https://github.com/sayakpaul/FunMatch-Distillation) as a guide.

The implementation of the d-Simplex was taken from [d-Simplex](https://github.com/NiccoBiondi/cores-compatibility/blob/main/src/cores/model.py)

To run knowledge distillation with the BitResNets models run knowledge_distillation.py. For example:

```
python3 knowledge_distillation.py  --model_teacher BiT-M-R152x2 --model_student BiT-M-R50x1  --name flowers_fixed --logdir distillation --finetuned bit_logs/flowers/bit.pth.tar --dataset oxford_flowers102 --datadir datadir --fixed_classifier True
```
To run knowledge distillation with the BitResNets models while testing the consistent teacher guideline run knowledge_distillation_experiments.py and change the teaching parameter (fix_cc, ind_rc or same_ic, there is no FunMatch since that is used in regular knowledge distillation). For example:

```
python3 knowledge_distillation_experiments.py  --model_teacher BiT-M-R152x2 --model_student BiT-M-R50x1  --name flowers_fixed_same_ic --logdir distillation --finetuned bit_logs/flowers/bit.pth.tar --dataset oxford_flowers102 --datadir datadir --fixed_classifier True --teaching same_ic
```

To run knowledge distillation with the ViTs models run knowledge_distillation_vit.py. For example:

```
python3 knowledge_distillation_vit.py  --model_teacher ViT-B_32 --model_student ViT-B_32  --name vit_flowers --logdir distillation --finetuned vit_logs/flowers102_checkpoint.bin --dataset oxford_flowers102 --datadir datadir --fixed_classifier True  
```

### Graphs

To replicate the graphs I put on the paper run graphs.py (use the same batch, training_size, epochs and eval_every of the run you want to turn into a graph):

```
python3 graphs.py --name graph_flower_comparison --log /distillation/flowers/train.log*/distillation/flowers_fixed/train.log --batch 512 --training_size 1020 --epochs 1000 --eval_every 250 
```
More runs on the same dataset and with the same hyperpameter can be plot on the same graph by concatenating their train.log paths with a "*".

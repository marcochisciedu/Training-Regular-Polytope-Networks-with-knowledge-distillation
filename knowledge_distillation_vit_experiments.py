import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import bit_pytorch.train as train
import torch
from collections import OrderedDict
import bit_pytorch.models as models
import argparse
import bit_hyperrule
import bit_common
import numpy as np
import bit_pytorch.lbtoolbox as lb
import matplotlib.pyplot as plt
import torchvision as tv
from os.path import join as pjoin
import math
from vit_models.modeling import VisionTransformer, CONFIGS

def argparser(known_models):
  parser = argparse.ArgumentParser(description="Knowledge distillation")
  parser.add_argument("--model_teacher", choices=list(known_models) + list(CONFIGS),
                      help="Which variant to use")
  parser.add_argument("--model_student", choices=list(known_models) + list(CONFIGS),
                      help="Which variant to use")
  parser.add_argument("--dataset", choices=list(bit_hyperrule.known_dataset_sizes.keys()),
                      help="Choose the dataset. It should be easy to add your own! "
                      "Don't forget to set --datadir if necessary.")
  parser.add_argument("--logdir", required=True,
                      help="Where to log training info (small).")
  parser.add_argument("--name", required=True,
                      help="Name of this run. Used for monitoring and checkpointing.")
  parser.add_argument("--finetuned", required=True,
                      help="Where to search for finetuned BiT/ViT models.")
  parser.add_argument("--batch", type=int, default=512,
                      help="Batch size.")
  parser.add_argument("--epochs", type=int, default=300,
                      help="Epochs")
  parser.add_argument("--temperature", type=int, default=2,
                      help="temperature")
  parser.add_argument("--weight_decay", type=float, default=1e-5,
                      help="weight decay")
  parser.add_argument("--warmup_steps", type=int, default=1500,
                      help="warmup steps")
  parser.add_argument("--clip_threshold", type=float, default=1,
                      help="clip threshold")
  parser.add_argument("--base_lr", type=float, default=0.001,
                      help="Base learning-rate.")
  parser.add_argument("--batch_split", type=int, default=1,
                      help="Number of batches to compute gradient on before updating weights.")
  parser.add_argument("--eval_every", type=int, default=None,
                      help="Run prediction on validation set every so many steps."
                      "Will always run one evaluation at the end of training.")
  parser.add_argument("--fixed_classifier", type=bool, default= False,
                      help= "True if the student model has a fixed classifier.")
  parser.add_argument("--student_pretained", type=bool, default= False,
                      help= "True if the student model is pretrained")
  parser.add_argument("--finetuned_student", 
                      help="Where to search for finetuned BiT/ViT model for the student.")
  return parser

def mixup(images, labels):
    alpha = torch.rand(1).to(images.device)
    mixedup_images = (alpha * images +
                     (1 - alpha) * torch.flip(images, dims=[0]))
    return mixedup_images, labels

def get_lr(learning_rate_base, total_steps, warmup_learning_rate, warmup_steps, step):
  if total_steps < warmup_steps:
    raise ValueError("Total_steps must be larger or equal to warmup_steps.")
  learning_rate = (0.5* learning_rate_base* (1 + torch.cos( math.pi
                    * (torch.tensor(step, dtype=torch.float32 ) - warmup_steps)
                    / float(total_steps - warmup_steps)
                )
            )
        )

  if warmup_steps > 0:
    if learning_rate_base < warmup_learning_rate:
      raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
    slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * torch.tensor(step, dtype=torch.float32 ) + warmup_learning_rate
    learning_rate = torch.where(torch.tensor(step < warmup_steps,dtype=torch.bool), warmup_rate, learning_rate)
    return torch.where(torch.tensor(step > total_steps,dtype=torch.bool), 0.0, learning_rate)
  
def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]

def run_eval(model, data_loader, device, chrono, logger, step, feat_dim=2048, num_classes=102):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      if step != "start":
        y= torch.nn.functional.pad(input=y, pad=(1, feat_dim+1-num_classes), mode='constant', value=0)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)[0]
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())


  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5

def get_datasets(args, logger):
  """Returns train and validation datasets."""
  train_tx = tv.transforms.Compose([
    tv.transforms.Resize((420, 420)),
    tv.transforms.RandomCrop((384, 384)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
    tv.transforms.Resize((384,384)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  elif args.dataset == 'oxford_flowers102':
    train_set = tv.datasets.Flowers102(args.datadir, transform=train_tx, split="train", download=True)
    valid_set = tv.datasets.Flowers102(args.datadir, transform=val_tx, split="val", download=True)
  elif args.dataset == "oxford_iiit_pet":
    train_set = tv.datasets.OxfordIIITPet(root='/andromeda/datasets/', split = 'trainval', transform = train_tx)
    valid_set = tv.datasets.OxfordIIITPet(root='/andromeda/datasets/', split = 'test', transform = val_tx)
  elif args.dataset == "food-101":
    train_set = tv.datasets.Food101(root='/andromeda/datasets/CoOp/', split= "train", transform= train_tx)
    valid_set = tv.datasets.Food101(root='/andromeda/datasets/CoOp/', split= "test", transform= val_tx)
  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader

def kl_divergence(true_p, q):
    true_prob = torch.nn.functional.softmax(true_p, dim= 1)
    loss_1 = -torch.nn.functional.cross_entropy(input=true_p, target=true_prob)
    loss_2 = torch.nn.functional.cross_entropy(input=q, target=true_prob)   
    loss = loss_1 + loss_2
    return loss

def visualize_lrs(logger, args, total_steps, warmup_learning_rate,warmup_steps):
  logger.info(f"Visualising learning rates in learning rates.png")
  lrs = [get_lr(learning_rate_base= args.base_lr , total_steps= total_steps, warmup_learning_rate=warmup_learning_rate, warmup_steps=warmup_steps,step= step) for step in range(total_steps)]
  plt.figure(1)
  plt.plot(lrs)
  plt.xlabel("Step", fontsize=14)
  plt.ylabel("LR", fontsize=14)
  plt.show()
  plt.savefig('learning rates.png')

def visualize_dataset(logger, train_loader):
  logger.info(f"Visualising some images of the dataset in dataset.png")
  sample_images, sample_labels = next(iter(train_loader))
  mixed_up_images, _ = mixup(sample_images, sample_labels)
  plt.figure(figsize=(10, 10), num=0)
  for n in range(25):
      ax = plt.subplot(5, 5, n + 1)
      plt.imshow(mixed_up_images[n].squeeze().permute(1,2,0))
      plt.axis("off")
  plt.show()
  plt.savefig('dataset.png')

def dsimplex( feat_dim, device='cuda'):
    def simplex_coordinates(n, device):
        t = torch.zeros((n + 1, n), device=device)
        torch.eye(n, out=t[:-1,:], device=device)
        val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
        t[-1,:].add_(val)
        t.add_(-torch.mean(t, dim=0))
        t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
        return t.cpu()

    ds = simplex_coordinates(feat_dim, device)
    return ds

def main(args):
    logger = logger = bit_common.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    chrono = lb.Chrono()

    train_set, valid_set, train_loader, valid_loader = get_datasets(args, logger)

    visualize_dataset(logger, train_loader)
    if args.dataset == "cifar10":
      num_classes=len(valid_set.classes)
    elif args.dataset == "oxford_flowers102":
      num_classes=102
    elif args.dataset == "oxford_iiit_pet":
      num_classes=37
    elif args.dataset == "food-101":
      num_classes=101

    config_teacher = CONFIGS[args.model_teacher]
    img_size = 384
    teacher = VisionTransformer(config_teacher, img_size, zero_head=True, num_classes=num_classes)
  
    checkpoint = torch.load(args.finetuned, map_location="cpu")
   
    logger.info(f"Loading teacher model from {args.finetuned}")
    teacher.load_state_dict(checkpoint)

    logger.info("Moving teacher model onto all GPUs")
    teacher = torch.nn.DataParallel(teacher)    

    teacher = teacher.to(device)
    for p in teacher.parameters():
      p.requires_grad = False
    teacher.eval()
    run_eval(teacher, valid_loader, device, chrono, logger, step='start')   #validating teacher model

    config_student = CONFIGS[args.model_student]
    student = VisionTransformer(config_student, img_size, zero_head=True, num_classes=num_classes)

    if args.student_pretained == True:
      logger.info(f"Loading student model from {args.finetuned_student}")
      student.load_from(np.load(args.finetuned_student))


    feat_dim = 2048
    if args.fixed_classifier is True:
      fixed_weights = dsimplex(feat_dim= feat_dim, device=device)
      student.head=torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(config_student.hidden_size, feat_dim, bias=False)),  
        ('fc2', torch.nn.Linear(feat_dim, feat_dim+1, bias=False)),
      ]))

      student.head.fc2.weight.requires_grad = False  # set no gradient for \ fixed classifier
      student.head.fc2.weight.copy_(fixed_weights)   # set the weights for the classifier

    logger.info("Moving student model onto all GPUs")
    student = torch.nn.DataParallel(student)

    run_eval(student, valid_loader, device, chrono, logger, step='first')

    student = student.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.base_lr, weight_decay= args.weight_decay)
    optimizer.zero_grad()

    total_steps = int(len(train_set)/ args.batch * args.epochs)
    warmup_steps = args.warmup_steps
    warmup_learning_rate = 0
    visualize_lrs(logger, args,total_steps,warmup_learning_rate, warmup_steps)
    
    student.train()
    step=0

    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    try:
      logger.info(f"Student model will be saved in '{savename}'")
      checkpoint = torch.load(savename, map_location="cpu")
      logger.info(f"Found saved student model to resume from at '{savename}'")

      step = checkpoint["step"]
      student.load_state_dict(checkpoint["model"])
      optimizer.load_state_dict(checkpoint["optim"])
      logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
      logger.info("Knowledge distillation")

    logger.info("Starting training!")
    accum_steps = 0


    for x, y in train.recycle(train_loader):
      #stop training if over.
      if step >= total_steps:
        break 
      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate
      lr = get_lr(learning_rate_base= args.base_lr , total_steps= total_steps, warmup_learning_rate=warmup_learning_rate, warmup_steps=warmup_steps,step= step)
      for param_group in optimizer.param_groups:
        param_group["lr"] = lr

      x, y = mixup(x,y)

      # compute output
      with torch.no_grad():
        teacher_prediction = teacher(x)[0]
        teacher_prediction= torch.nn.functional.pad(input=teacher_prediction, pad=(1, feat_dim+1-num_classes), mode='constant', value=0)
      student_prediction = student(x)[0]
      c = kl_divergence(teacher_prediction/args.temperature ,student_prediction/args.temperature)
      c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      (c / args.batch_split).backward()
      accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}/{total_steps}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  
      logger.flush()

      # Update params
      if accum_steps == args.batch_split:
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_threshold)

        optimizer.step()
        optimizer.zero_grad()
        step += 1
        accum_steps = 0

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(student, valid_loader, device, chrono, logger, step)
          if args.save:
              torch.save({
                  "step": step,
                  "model": student.state_dict(),
                  "optim" : optimizer.state_dict(),
              }, savename)

    #validating student model at the end of training
    run_eval(student, valid_loader, device, chrono, logger, step='end')
    if args.save:
              torch.save({
                  "step": step,
                  "model": student.state_dict(),
                  "optim" : optimizer.state_dict(),
              }, savename)


if __name__ == "__main__":
  parser = argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  main(parser.parse_args())
import argparse
import matplotlib.pyplot as plt
import os

def argparser():
  parser = argparse.ArgumentParser(description="Creating graphs")
  parser.add_argument("--name", required=True,
                      help="Name of this run. Used for monitoring and checkpointing.")
  parser.add_argument("--log", required=True,
                      help="Paths of the training logs"
                      "path*path")
  parser.add_argument("--batch", type=int, default=512,
                      help="Batch size.")
  parser.add_argument("--training_size", type=int, default=50000,
                      help="Training set size.")                    
  parser.add_argument("--epochs", type=int, default=300,
                      help="Epochs")
  parser.add_argument("--eval_every", type=int, default=None,
                      help="Run prediction on validation set every so many steps."
                      "Will always run one evaluation at the end of training.")
  return parser

def read_log(log, args):
  with open(log, 'r') as txt:
      list_of_lines = txt.readlines()
  val_steps=[]
  val_losses=[]
  val_top1=[]
  train_steps=[]
  train_losses=[]
  for line in list_of_lines:
      if "top1" in line.split(" "):
        if "start" not in line:
          val_steps.append(''.join(line.split("Validation@")[1].split(" loss")[0]))
          val_losses.append(float(''.join(line.split("loss ")[1].split(", top1")[0])))
          val_top1.append(float(''.join(line.split("top1 ")[1].split("%, top5")[0])))
      if "[step" in line.split(" ") and int(''.join(line.split("[step ")[1].split("/")[0]))%500 ==0:  #save a value every 500 steps
        if ''.join(line.split("[step ")[1].split("/")[0]) not in train_steps:
          train_steps.append(''.join(line.split("[step ")[1].split("/")[0]))
          train_losses.append(float(''.join(line.split("loss=")[1].split(" (lr=")[0])))
  total_steps = int(args.training_size/ args.batch * args.epochs)
  val_steps[-1]=total_steps  #end=total_steps
  val_steps= list(map(lambda x:int(x)/(args.training_size/args.batch) ,val_steps))

  train_steps= list(map(lambda x:int(x)/(args.training_size/args.batch) ,train_steps))

  return val_steps, val_losses, val_top1, train_steps, train_losses
def main(args):
    num_logs = (args.log.count('*'))+1
    log = args.log.split("*")
    total_val_steps=[]
    total_val_losses =[]
    total_val_top1 =[]
    total_train_steps=[]
    total_train_losses=[]
    for i in range(num_logs):
      val_steps, val_losses, val_top1, train_steps, train_losses= read_log(log[i], args)
      total_val_steps.append(val_steps)
      total_val_losses.append(val_losses)
      total_val_top1.append(val_top1)
      total_train_steps.append(train_steps)
      total_train_losses.append(train_losses)


    names = ["ViT 10000 epochs"]
    assert len(names)==num_logs
    plt.figure(figsize=(18,5))
    plt.subplot(131)
    for i in range(num_logs):
      plt.plot(total_train_steps[i], total_train_losses[i], label=names[i])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("training loss")

    plt.subplot(132)
    for i in range(num_logs):
      plt.plot(total_val_steps[i], total_val_losses[i], label=names[i])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("validation loss")

    plt.subplot(133)
    for i in range(num_logs):
      plt.plot(total_val_steps[i], total_val_top1[i], label=names[i])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("top 1 validation accuracy %")

    plt.show()
    plt.savefig(args.name)


if __name__ == "__main__":
  parser = argparser()
  main(parser.parse_args())
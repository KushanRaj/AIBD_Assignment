import os
import torch
import numpy as np
from datetime import datetime
import time


def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")


def add_args(parser):
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [default: output/year-month-date_hour-minute]",
    )
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed")
    parser.add_argument("--dist", action="store_true", help="start distributed training")
    parser.add_argument("--dset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading"
    )
    parser.add_argument("--kernel_size", type = int, default = 2, help="Kernel size")
    parser.add_argument("--hid_dim", type = int, default = 32, help="Kernel size")
    parser.add_argument(
        "--init_residue", type=bool, default=False, help="init residual block to identity"
    )
    #parser.add_argument("--lr", type = float, default = 1e-3, help="Learning Rate")
    parser.add_argument("--optim", type=str, default="sgd", help="optimizer name")
    parser.add_argument("--lr", type=float, default=0.001, help="sgd learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd optimizer momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="sgd optimizer weight decay"
    )
    parser.add_argument("--nesterov", type=bool, default=False, help="nesterov in sgd optim")
    parser.add_argument("--loss_scale", type = float, default = 1, help="Learning Rate")
    parser.add_argument(
        "--lr_step_mode",
        type=str,
        default="epoch",
        help="choose lr step mode, choose one of [epoch, step]",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="lr warmup in epochs/steps based on epoch step mode"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr_sched", type=str, default="cosine", help="lr scheduler name")
    parser.add_argument(
        "--lr_decay_steps",
        type=str,
        default="100,150",
        help="multi step lr scheduler milestones",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.1, help="multi step lr scheduler decay gamma"
    )
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--aug", action="store_true", help="Use augmentation")
    parser.add_argument("--wandb", action="store_true", help="start wandb logging")
    parser.add_argument("--eval_every", type=int, default=1, help="eval frequency")
    parser.add_argument("--log_every", type=int, default=1, help="logging frequency")

    return parser


def setup_device(dist):
    if dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def pbar(p=0, msg="", bar_len=20):
    msg = msg.ljust(50)
    block = int(round(bar_len * p))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="")
    if p == 1:
        print()


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


class RunningMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        attr_keys = list(self.__dict__.keys())
        for key in attr_keys:
            delattr(self, key)
        self.cntr = 0

    def add(self, **kwargs):
        for key, value in kwargs.items():
            attr = getattr(self, key, None)
            if attr is None and self.cntr == 0:
                setattr(self, key, value)
            elif attr is None:
                raise ValueError(f"invalid key: {key}")
            else:
                attr = attr + value
                setattr(self, key, attr)
        self.cntr += 1

    def get(self):
        return {
            key: value.item() / self.cntr for key, value in self.__dict__.items() if key != "cntr"
        }

    def msg(self):
        avg_metrics = {
            key: value.item() / self.cntr for key, value in self.__dict__.items() if key != "cntr"
        }
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


class TimeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.sample_cnt = 0

    def add(self, n_samples=1):
        self.sample_cnt += n_samples

    def get(self):
        return int(self.sample_cnt / (time.time() - self.start))

    def msg(self):
        return f"samples/sec: {self.get()}"
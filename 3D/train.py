import argparse
import utils
import random
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, DistributedSampler
import network
import os
from torch.nn.parallel import DistributedDataParallel
import wandb
import json


class Trainer:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        local_rank = 0
        self.main_thread = True if local_rank == 0 else False
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        if "cifar" in args.dset:
            if args.aug:
                t = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                t = []
            norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        else:
            raise NotImplementedError(f"args.dset = {args.dset} is not implemented")

        train_transform = transforms.Compose(
            [
                *t,
                transforms.ToTensor(),
                norm,
            ]
        )
        
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                norm,
            ]
        )

        if args.dset == "cifar10":
            train_dset = datasets.CIFAR10(
                root=args.data_root, train=True, transform=train_transform, download=True
            )
            val_dset = datasets.CIFAR10(
                root=args.data_root, train=False, transform=val_transform, download=True
            )
            n_cls = 10
        elif args.dset == "cifar100":
            train_dset = datasets.CIFAR100(
                root=args.data_root, train=True, transform=train_transform, download=True
            )
            val_dset = datasets.CIFAR100(
                root=args.data_root, train=False, transform=val_transform, download=True
            )
            n_cls = 100
        else:
            raise NotImplementedError(f"args.dset = {args.dset} is not implemented")

        if self.main_thread:
            print(f"setting up dataset, train: {len(train_dset)}, val: {len(val_dset)}")

        if args.dist:
            train_sampler = DistributedSampler(train_dset)
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
            )

        self.val_loader = DataLoader(
            val_dset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
        )

        hid_dims = [int(i) for i in args.hid_dims.split(',')]

        if "cifar" in args.dset:
            model = network.Model(1, n_cls, hid_dims, args.n_levels)
        else:
            raise NotImplementedError(f"args.dset = {args.dset} is not implemented")
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(self.device)
        if self.main_thread:
            print(f"# of model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        self.criterion = torch.nn.CrossEntropyLoss()
        if args.optim == "sgd":
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        elif args.optim == "Adam":
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=args.lr,
            )
        else:
            raise ValueError(f"args.optim_type = {args.optim} not implemented")

        if self.args.lr_step_mode == "epoch":
            total_steps = args.epochs - args.warmup
        else:
            total_steps = int(args.epochs * len(self.train_loader) - args.warmup)
        if args.warmup > 0:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 * group["lr"]
        if args.lr_sched == "cosine":
            self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, total_steps)
        elif args.lr_sched == "multi_step":
            milestones = [
                int(milestone) - total_steps for milestone in args.lr_decay_steps.split(",")
            ]
            self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=milestones, gamma=args.lr_decay
            )
        else:
            raise ValueError(f"args.lr_sched_type = {args.lr_sched} not implemented")

        if os.path.exists(os.path.join(self.out_dir, "last.ckpt")):
            if args.resume == False and self.main_thread:
                raise ValueError(
                    f"directory {self.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(self.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"loaded checkpoint, resuming training expt from {self.start_epoch} to {args.epochs} epochs."
                )
        else:
            if args.resume == True and self.main_thread:
                raise ValueError(
                    f"resume training args are true but no checkpoint found in {self.out_dir}"
                )
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.start_epoch = 0
            if self.main_thread:
                print(f"starting fresh training expt for {args.epochs} epochs.")
        self.train_steps = self.start_epoch * len(self.train_loader)

        self.log_wandb = False
        self.metric_meter = utils.AvgMeter()
        self.time_meter = utils.TimeMeter()
        if self.main_thread:
            self.log_f = open(os.path.join(self.out_dir, "logs.txt"), "w")
            print(f"start file logging @ {os.path.join(self.out_dir, 'logs.txt')}")
            if args.wandb:
                self.log_wandb = True
                run = wandb.init()
                print(f"start wandb logging @ {run.get_url()}")
                self.log_f.write(f"\nwandb url @ {run.get_url()}\n")

    def train_epoch(self):
        self.metric_meter.reset()
        self.time_meter.reset()
        self.model.train()
        for indx, (img, target) in enumerate(self.train_loader):
            img, target = img.to(self.device), target.to(self.device)

            img = img.unsqueeze(1)

            pred = self.model(img)
            loss = self.criterion(pred, target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"train_loss": loss.item(), "train_acc": acc}
            self.metric_meter.add(metrics)
            self.time_meter.add(
                img.shape[0] * torch.cuda.device_count() if self.args.dist else img.shape[0]
            )

            if self.main_thread and indx % self.args.log_every == 0:
                if self.log_wandb:
                    wandb.log({"train step": self.train_steps, "train loss": loss.item()})
                utils.pbar(
                    indx / len(self.train_loader),
                    msg=self.metric_meter.msg() + self.time_meter.msg(),
                )

            if self.args.lr_step_mode == "step":
                if self.train_steps < self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = (
                        self.train_steps / (self.args.warmup) * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            self.train_steps += 1
        if self.main_thread:
            utils.pbar(1, msg=self.metric_meter.msg() + self.time_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.time_meter.reset()
        self.model.eval()
        for indx, (img, target) in enumerate(self.val_loader):
            img, target = img.to(self.device), target.to(self.device)

            img = img.unsqueeze(1)

            pred = self.model(img)
            loss = self.criterion(pred, target)

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"val_loss": loss.item(), "val_acc": acc}
            self.metric_meter.add(metrics)
            self.time_meter.add(img.shape[0])
            utils.pbar(
                indx / len(self.val_loader), msg=self.metric_meter.msg() + self.time_meter.msg()
            )
        utils.pbar(1, msg=self.metric_meter.msg() + self.time_meter.msg())

    def train(self):
        best_train, best_val = 0, 0
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(
                    f"epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                )
                print("---------------")

            self.train_epoch()
            if self.main_thread:
                train_metrics = self.metric_meter.get()
                if train_metrics["train_acc"] > best_train:
                    print(
                        "\x1b[34m"
                        + f"train acc improved from {round(best_train, 5)} to {round(train_metrics['train_acc'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train_acc"]
                msg = f"epoch: {epoch}, last train: {round(train_metrics['train_acc'], 5)}, best train: {round(best_train, 5)}"

                val_metrics = {}
                if epoch % self.args.eval_every == 0:
                    self.eval()
                    val_metrics = self.metric_meter.get()
                    if val_metrics["val_acc"] > best_val:
                        print(
                            "\x1b[33m"
                            + f"val acc improved from {round(best_val, 5)} to {round(val_metrics['val_acc'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val_acc"]
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.args.out_dir, f"best.ckpt"),
                        )
                    msg += f", last val: {round(val_metrics['val_acc'], 5)}, best val: {round(best_val, 5)}"

                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                self.log_f.flush()

                if self.log_wandb:
                    train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                    val_metrics = {"epoch " + key: value for key, value in val_metrics.items()}
                    wandb.log(
                        {
                            "epoch": epoch,
                            **train_metrics,
                            **val_metrics,
                            "lr": self.optim.param_groups[0]["lr"],
                        }
                    )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )

            if self.args.lr_step_mode == "epoch":
                if epoch <= self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = epoch / self.args.warmup * self.args.lr
                else:
                    self.lr_sched.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = utils.add_args(parser)
    args = parser.parse_args()
    utils.print_args(args)

    trainer = Trainer(args)
    trainer.train()

    if args.dist:
        torch.distributed.destroy_process_group()
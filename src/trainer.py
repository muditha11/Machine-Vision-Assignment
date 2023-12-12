import os
from omegaconf import OmegaConf
import yaml
from .utils import init_obj_from_conf, Logger, get_nested_attr
from torch.utils.data import DataLoader, Subset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm.auto import tqdm
import torch


class Trainer:
    def _load_data(self, mock_batch_count):
        train_ds = init_obj_from_conf(self.conf.data, split="train")
        if mock_batch_count > 0:
            train_ds = Subset(
                train_ds,
                range(int(mock_batch_count * self.conf.train.loader_params.batch_size)),
            )
        self.train_dl: DataLoader = DataLoader(
            train_ds, **dict(self.conf.train.loader_params)
        )
        if "val" in self.conf:
            val_ds = init_obj_from_conf(self.conf.data, split="val")
            if mock_batch_count > 0:
                val_ds = Subset(
                    val_ds,
                    range(
                        int(mock_batch_count * self.conf.val.loader_params.batch_size)
                    ),
                )
            self.val_dl: DataLoader = DataLoader(
                val_ds, **dict(self.conf.train.loader_params)
            )
            self.do_val: bool = True
        else:
            self.val_dl: DataLoader = None
            self.do_val: bool = False

    def _load_train_objs(self):
        # model
        self.model: nn.Module = init_obj_from_conf(self.conf.model, device=self.device)
        if "freeze" in self.conf.model:
            for net_key in self.conf.model.freeze:
                self.logger.info(f"Feezing subnetwork '{net_key}' of the model")
                net = get_nested_attr(self.model, net_key)
                for para in net.parameters():
                    para.requires_grad = False

        # loss
        self.loss_fn: nn.Module = init_obj_from_conf(
            self.conf.loss, device=self.device, logger=self.logger
        )

        # optimizer
        if "optimizer" in self.conf:
            self.optimizer: Optimizer = init_obj_from_conf(
                self.conf.optimizer, params=self.model.parameters()
            )
        else:
            self.optimizer: Optimizer = Adam(self.model.parameters())

        # lr scheduler
        if "lr_scheduler" in self.conf:
            self.lr_scheduler: LRScheduler = init_obj_from_conf(
                self.conf.lr_scheduler, optimizer=self.optimizer
            )
        else:
            self.lr_scheduler: LRScheduler = None

    def __init__(self, config_path, device, mock_batch_count=-1) -> None:
        with open(config_path) as handler:
            conf = OmegaConf.create(yaml.load(handler, yaml.FullLoader))
        self.conf = conf
        self.device = device

        self.logger = Logger()
        self._load_data(mock_batch_count)
        self._load_train_objs()

    def _train_loop(self, epoch: int) -> float:
        losses = []
        with tqdm(
            total=len(self.train_dl),
            disable=not self.logger.display_info,
            desc="Training",
        ) as pbar:
            for batch in self.train_dl:
                self.optimizer.zero_grad()
                info = self.model(batch)
                loss = self.loss_fn(info, batch)
                loss.backward()
                self.optimizer.step()
                pbar.update()
                losses.append(loss.cpu().detach().item())
                pbar.set_postfix(loss=losses[-1])
        train_loss = sum(losses) / len(losses)
        return train_loss

    def _val_loop(self, epoch: int) -> float:
        losses = []
        with tqdm(
            total=len(self.val_dl),
            disable=not self.logger.display_info,
            desc="Validating",
        ) as pbar:
            with torch.no_grad():
                for batch in self.val_dl:
                    self.optimizer.zero_grad()
                    info = self.model(batch)
                    loss = self.loss_fn(info, batch)
                    pbar.update()
                    losses.append(loss.cpu().detach().item())
                    pbar.set_postfix(loss=losses[-1])
        val_loss = sum(losses) / len(losses)
        return val_loss

    def _init_fit(self, out_root: str) -> str:
        class_out = os.path.join(out_root, self.conf.name)
        i = 0
        while os.path.exists(os.path.join(class_out, f"run{i}")):
            i += 1
        out_dir = os.path.join(class_out, f"run{i}")
        os.makedirs(os.path.join(out_dir, "ckpts"))
        self.logger.init_plotter(out_dir)
        with open(os.path.join(out_dir, "configuration.yaml"), "w") as handler:
            yaml.dump(OmegaConf.to_container(self.conf))

        return out_dir

    def fit(self, out_root) -> None:
        out_dir = self._init_fit(out_root)

        history = []
        best_epoch = -1
        best_loss = float("inf")
        best_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        for epoch in range(self.conf.epochs):
            self.logger.info(
                f"----EPOCH {str(epoch+1).rjust(len(str(self.conf.epochs)), '0')}/{self.conf.epochs}----"
            )
            train_loss = self._train_loop(epoch)
            val_loss = self._val_loop(epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            history.append([train_loss, val_loss])
            self.logger.accumulate_train_loss(train_loss)
            self.logger.accumulate_val_loss(val_loss)
            self.logger.step(epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
            else:
                if epoch - best_epoch > self.conf.train.tollerance:
                    self.logger.info(
                        f"Stopping early. Best epoch found at epoch {best_epoch+1}"
                    )
                    break
        final_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(
            {
                "loss": best_loss,
                "epoch": best_epoch,
                "state": best_state,
            },
            os.path.join(out_dir, "ckpts", "best.tar"),
        )

        torch.save(
            {
                "loss": val_loss,
                "epoch": epoch,
                "state": final_state,
            },
            os.path.join(out_dir, "ckpts", "final.tar"),
        )

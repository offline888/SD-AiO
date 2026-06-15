import argparse
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from src.data.dataset import InterleavedShuffleDataset, MultiLabelClassification
from src.networks.degnet import DegNet_CLIP, DegNet_DINO
from torch.utils.data import ChainDataset, DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelJaccardIndex,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

MODEL_FACTORY = {"DegNet_CLIP": DegNet_CLIP, "DegNet_DINO": DegNet_DINO}


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = OmegaConf.load(args.config)

        self.exp_name = self.config.get("name", "default")
        self.exp_name = f"{self.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        set_seed(self.config.seed)

        self.accelerator: Optional[Accelerator] = None
        self.is_main: bool = False
        self.unwrap_fn: Optional[callable] = None

        self.dataloaders: Dict[str, Optional[DataLoader]] = {}
        self.datasets: Dict[str, List[Any]] = {}
        self.sampler = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.metrics: Optional[Dict[str, Any]] = None

        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_mAP: float = 0.0

        self.logger = logging.getLogger(__name__)

    def setup_dist(self) -> None:
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.config.accelerator.get(
                "find_unused_parameters", False
            )
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.accelerator.get("grad_accum", 1),
            mixed_precision=self.config.accelerator.get("mixed_precision", "no"),
            log_with="swanlab" if self.config.logging.use_swanlab else None,
            project_dir=self.config.experiments_dir,
            kwargs_handlers=[ddp_kwargs],
        )

        self.is_main = self.accelerator.is_main_process
        self.unwrap_fn = self.accelerator.unwrap_model

    def init_logger(self) -> None:
        self.ckpt_dir = os.path.join(
            self.config.experiments_dir, self.exp_name, "checkpoints"
        )

        if self.is_main and self.config.logging.use_swanlab:
            self.log_dir = os.path.join(
                self.config.experiments_dir, self.exp_name, "logs"
            )
            swanlab_config = OmegaConf.to_container(self.config)
            swanlab_config["log_dir"] = self.log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            self.accelerator.init_trackers(
                self.config.logging.swanlab_project, swanlab_config
            )

        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def build_dataloader(self) -> None:
        train_datasets, val_datasets = [], []
        deg_types = self.config.data.degradations

        for data_name, data_config in self.config.data.datasets.items():
            resize_size = data_config.get("resize", 224)
            ops = [
                transforms.Resize((resize_size, resize_size)),
            ]
            if data_config.get("use_hflip"):
                ops.append(transforms.RandomHorizontalFlip())
            if data_config.get("use_rot"):
                ops.append(transforms.RandomRotation(15))
            ops.extend(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            dataset_transforms = transforms.Compose(ops)
            dataset = MultiLabelClassification(
                data_config, deg_types, dataset_transforms
            )

            if data_name.startswith("ValDataset"):
                val_datasets.append(dataset)
            elif data_name.startswith("TrainDataset"):
                train_datasets.append(dataset)
            else:
                raise ValueError(
                    f"Invalid dataset name: '{data_name}'. "
                    "Must start with 'TrainDataset' or 'ValDataset'."
                )

        self.datasets["train"] = train_datasets
        self.datasets["val"] = val_datasets

        val_cfg = self.config.data.dataloader.val
        val_loader = (
            DataLoader(
                ChainDataset(val_datasets),
                batch_size=val_cfg.batch_size,
                num_workers=val_cfg.get("num_workers", 4),
                pin_memory=val_cfg.get("pin_memory", True),
                persistent_workers=val_cfg.get("persistent_workers", True),
                drop_last=False,
                prefetch_factor=val_cfg.get("prefetch_factor", 2),
            )
            if val_datasets
            else None
        )

        train_cfg = self.config.data.dataloader.train
        train_loader = DataLoader(
            InterleavedShuffleDataset(
                train_datasets, buffer_size=3000, seed=self.config.seed
            ),
            shuffle=False,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.get("num_workers", 8),
            pin_memory=train_cfg.get("pin_memory", True),
            persistent_workers=train_cfg.get("persistent_workers", True),
            drop_last=train_cfg.get("drop_last", True),
            prefetch_factor=train_cfg.get("prefetch_factor", 4),
        )

        self.dataloaders["train"] = train_loader
        self.dataloaders["val"] = val_loader

    def build_model(self) -> None:
        Model = MODEL_FACTORY[self.config.network.type]
        backbone_key = (
            "clip_type" if "CLIP" in self.config.network.type else "dino_type"
        )

        self.model = Model(
            feature_dim=self.config.network.get("feature_dim", 512),
            num_types=self.config.network.num_classes,
            freeze_encoder=self.config.network.get("freeze_encoder", False),
            patch_size=self.config.network.get("patch_size", 14),
            encoder_layer_index=self.config.network.get("encoder_layer_index", -1),
            **{backbone_key: self.config.network.backbone},
        )

        num_labels = self.config.network.num_classes
        self.metrics = {
            "mAP": MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
            "F1": MultilabelF1Score(num_labels=num_labels, average="macro"),
            "Acc": MultilabelAccuracy(num_labels=num_labels, average="macro"),
            "AUROC": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "Precision": MultilabelPrecision(num_labels=num_labels, average="macro"),
            "Recall": MultilabelRecall(num_labels=num_labels, average="macro"),
            "IoU": MultilabelJaccardIndex(num_labels=num_labels, average="macro"),
        }

    def setup_optimization(self) -> None:
        opt_cls = getattr(torch.optim, self.config.train.optim.type)
        self.optimizer = opt_cls(
            self.model.parameters(),
            lr=self.config.train.optim.lr,
            weight_decay=self.config.train.optim.get("weight_decay", 0),
        )

        total_steps = (
            len(self.dataloaders["train"])
            // self.accelerator.gradient_accumulation_steps
        ) * self.config.train.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.train.scheduler.eta_min,
        )

        if self.dataloaders.get("val") is not None:
            (
                self.model,
                self.optimizer,
                self.dataloaders["train"],
                self.scheduler,
                self.dataloaders["val"],
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.dataloaders["train"],
                self.scheduler,
                self.dataloaders["val"],
            )
        else:
            self.model, self.optimizer, self.dataloaders["train"], self.scheduler = (
                self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    self.dataloaders["train"],
                    self.scheduler,
                )
            )

        for metric in self.metrics.values():
            metric.to(self.accelerator.device)

    def training_step(self, data: tuple) -> torch.Tensor:
        images, labels = data

        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                logits = self.model(images)
                probs = torch.softmax(logits, dim=-1)
                loss = F.binary_cross_entropy(probs, labels, reduction="sum")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.train.get("max_grad_norm", 1.0)
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                self.global_step += 1
                log_interval = self.config.logging.get("log_interval", 100)
                if self.global_step % log_interval == 0 and self.is_main:
                    self.accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                    )

        return loss

    @torch.no_grad()
    def validation(self) -> Dict[str, float]:
        for metric in self.metrics.values():
            metric.reset()

        self.model.eval()
        for imgs, lbls in self.dataloaders["val"]:
            with self.accelerator.autocast():
                logits = self.model(imgs)
                probs = torch.softmax(logits, dim=-1)[:, :, 0]
                lbls_multilabel = lbls[:, :, 0].long()

                gathered_probs = self.accelerator.gather_for_metrics(probs)
                gathered_lbls = self.accelerator.gather_for_metrics(lbls_multilabel)

                for metric in self.metrics.values():
                    metric.update(gathered_probs, gathered_lbls)

        results = {
            name: metric.compute().item() for name, metric in self.metrics.items()
        }
        self.model.train()

        return results

    def save_ckpt(self) -> None:
        torch.save(
            self.unwrap_fn(self.model).state_dict(),
            os.path.join(self.ckpt_dir, f"iter_{self.global_step}.pth"),
        )

    def train(self) -> None:
        self.setup_dist()
        self.init_logger()
        self.build_dataloader()
        self.build_model()
        self.setup_optimization()

        self.model.train()

        for epoch in range(self.config.train.num_epochs):
            self.current_epoch = epoch + 1
            pbar = tqdm(
                self.dataloaders["train"],
                disable=not self.is_main,
                desc=f"Epoch {self.current_epoch}",
            )

            for images, labels in pbar:
                loss = self.training_step((images, labels))

                if self.is_main:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )

            val_freq = self.config.val.get("val_freq", 1)
            if self.dataloaders["val"] is not None and (epoch + 1) % val_freq == 0:
                results = self.validation()

                if self.is_main:
                    metric_str = " | ".join(
                        [f"{m}: {results[m]:.4f}" for m in self.metrics.keys()]
                    )
                    print(f"\n{'=' * 60}")
                    print(f"Epoch {self.current_epoch} Validation")
                    print(f"{'=' * 60}")
                    print(metric_str)
                    print(f"{'=' * 60}")

                    self.accelerator.log(
                        {f"val/{m}": results[m] for m in self.metrics.keys()},
                        step=self.global_step,
                    )

                    if results["mAP"] > self.best_mAP:
                        self.best_mAP = results["mAP"]
                        torch.save(
                            self.unwrap_fn(self.model).state_dict(),
                            os.path.join(self.ckpt_dir, "best_model.pth"),
                        )
                        print(f"New Best mAP: {self.best_mAP:.4f}")

            save_freq = self.config.train.get("save_freq", 1)
            if self.is_main and (epoch + 1) % save_freq == 0:
                self.save_ckpt()

        self.close_logger()

    def close_logger(self) -> None:
        self.accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

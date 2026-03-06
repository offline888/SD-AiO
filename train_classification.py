import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
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


def train(args):

    config = OmegaConf.load(args.config)
    set_seed(config.seed)

    cudnn.benchmark = True

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.accelerator.get("find_unused_parameters", False)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.accelerator.get("grad_accum", 1),
        mixed_precision=config.accelerator.get("mixed_precision", "no"),
        log_with="swanlab" if config.logging.use_swanlab else None,
        project_dir=config.experiments_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    Model = MODEL_FACTORY[config.network.type]
    backbone_key = "clip_type" if "CLIP" in config.network.type else "dino_type"
    model = Model(
        feature_dim=config.network.feature_dim,
        num_types=config.network.num_classes,
        freeze_encoder=config.network.freeze_encoder,
        freeze_deg_dict=config.network.get("freeze_deg_dict", False),
        **{backbone_key: config.network.backbone},
    )

    train_datasets, val_datasets = [], []
    deg_types = config.data.degradations

    for data_name, data_config in config.data.datasets.items():
        ops = [
            transforms.Resize(
                (data_config.get("resize", 224), data_config.get("resize", 224))),
        ]
        if data_config.get("use_hflip"):
            ops.append(transforms.RandomHorizontalFlip())
        if data_config.get("use_rot"):
            ops.append(transforms.RandomRotation(15))
        ops.extend(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        dataset_transforms = transforms.Compose(ops)
        dataset = MultiLabelClassification(data_config, deg_types, dataset_transforms)

        if data_name.startswith("ValDataset"):
            val_datasets.append(dataset)
        elif data_name.startswith("TrainDataset"):
            train_datasets.append(dataset)
        else:
            raise ValueError(f"Invalid dataset name: {data_name}")

    # bulid
    val_cfg = config.data.dataloader.val
    val_loader = DataLoader(
        ChainDataset(val_datasets),
        batch_size=val_cfg.batch_size,
        num_workers=val_cfg.get("num_workers", 4),
        pin_memory=val_cfg.get("pin_memory", True),
        persistent_workers=val_cfg.get("persistent_workers", True),
        drop_last=False,
        prefetch_factor=val_cfg.get("prefetch_factor", 2),
    ) if val_datasets else None

    # Build train loader
    train_cfg = config.data.dataloader.train
    train_loader = DataLoader(
        InterleavedShuffleDataset(train_datasets, buffer_size=2000, seed=config.seed),
        shuffle=False,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.get("num_workers", 8),
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=train_cfg.get("persistent_workers", True),
        drop_last=train_cfg.get("drop_last", True),
        prefetch_factor=train_cfg.get("prefetch_factor", 2),
    )

    # Build optimizer and scheduler
    opt_cls = getattr(torch.optim, config.train.optim.type)
    optimizer = opt_cls(
        model.parameters(),
        lr=config.train.optim.lr,
        weight_decay=config.train.optim.weight_decay,
    )

    total_steps = (
        len(train_loader) // accelerator.gradient_accumulation_steps
    ) * config.train.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.train.scheduler.eta_min
    )

    # Pre-cache criterion
    criterion = getattr(nn, config.train.loss.type)()

    # Pre-create metrics on device
    num_labels = config.network.num_classes
    metrics = {
        "mAP": MultilabelAveragePrecision(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "F1": MultilabelF1Score(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "Acc": MultilabelAccuracy(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "AUROC": MultilabelAUROC(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "Precision": MultilabelPrecision(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "Recall": MultilabelRecall(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
        "IoU": MultilabelJaccardIndex(num_labels=num_labels, average="macro").to(
            accelerator.device
        ),
    }

    # Validation function
    def validate():
        """Validate on merged val loader. All classes have pos/neg samples → macro avg valid."""
        for metric in metrics.values():
            metric.reset()

        for imgs, lbls in val_loader:
            with torch.no_grad(), accelerator.autocast():
                _, probs, _ = model(imgs)
                gathered_probs = accelerator.gather_for_metrics(probs)
                gathered_lbls = accelerator.gather_for_metrics(lbls)
                for metric_name, metric in metrics.items():
                    metric.update(gathered_probs, gathered_lbls.long())

        return {name: metric.compute().item() for name, metric in metrics.items()}

    log_interval = config.logging.get("log_interval", 100)
    max_grad_norm = config.train.max_grad_norm
    is_main = accelerator.is_main_process

    if val_loader is not None:
        model, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(
            model, optimizer, train_loader, scheduler, val_loader
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    unwrap_fn = accelerator.unwrap_model

    if is_main and config.logging.use_swanlab:
        swanlab_config = OmegaConf.to_container(config)
        log_dir = config.logging.get("swanlab_log_dir", config.experiments_dir)
        swanlab_config["log_dir"] = log_dir
        os.makedirs(log_dir, exist_ok=True)
        accelerator.init_trackers(config.logging.swanlab_project, swanlab_config)

    best_mAP = 0
    global_step = 0
    ckpt_dir = os.path.join(config.experiments_dir, "checkpoints")

    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config.train.num_epochs):
        model.train()
        pbar = tqdm(train_loader, disable=not is_main, desc=f"Epoch {epoch + 1}")

        for images, labels in pbar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    _, _, logits = model(images)
                    loss = criterion(logits, labels)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    if global_step % log_interval == 0:
                        accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": optimizer.param_groups[0]["lr"],
                            },
                            step=global_step,
                        )

            if is_main:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        # Validation: single merged pass over all val datasets
        if val_loader is not None and (epoch + 1) % config.val.val_freq == 0:
            model.eval()
            results = validate()

            if is_main:
                metric_str = " | ".join([f"{m}: {results[m]:.4f}" for m in metrics.keys()])
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch + 1} Validation (all datasets merged)")
                print(f"{'=' * 60}")
                print(f"{metric_str}")
                print(f"{'=' * 60}")

                accelerator.log(
                    {f"val/{m}": results[m] for m in metrics.keys()},
                    step=global_step,
                )

                # Save best model based on mAP
                if results["mAP"] > best_mAP:
                    best_mAP = results["mAP"]
                    torch.save(
                        unwrap_fn(model).state_dict(),
                        os.path.join(ckpt_dir, "best_model.pth"),
                    )
                    print(f"New Best mAP: {best_mAP:.4f} | Saved to {ckpt_dir}/best_model.pth")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    train(parser.parse_args())
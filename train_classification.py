import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm

from src.networks.degnet import DegNet_CLIP, DegNet_DINO
from src.data.dataset import MultiLabelClassification

MODEL_FACTORY = {
    "DegNet_CLIP": DegNet_CLIP,
    "DegNet_DINO": DegNet_DINO
}

def transform_pipeline(data_config):
    resize = data_config.get('resize', 384)
    use_hflip = data_config.get('use_hflip', False)
    use_rot = data_config.get('use_rot', False)
    
    operations = [
        transforms.Resize((resize,resize)),
        transforms.RandomHorizontalFlip() if use_hflip else None,
        transforms.RandomRotation(15) if use_rot else None,
    ]
    operations = [op for op in operations if op is not None]
     
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return transforms.Compose(operations)

def train(args):

    config = OmegaConf.load(args.config)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.get('find_unused_parameters', False)
    )
    
    accelerator = Accelerator(
        mixed_precision=config.train.mixed_precision,
        log_with="swanlab" if config.logging.use_swanlab else None, 
        project_dir=config.name,
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process and config.logging.use_swanlab:
        accelerator.init_trackers(
            project_name=config.logging.swanlab_project, 
            config=OmegaConf.to_container(config, resolve=True),
        )
        print(f"[Info] Experiment started: {config.name}")

    set_seed(config.seed)

    data_list = []
    deg_types = config.data.degradations
    
    for data_name, data_config in config.data.datasets.items():
        try:
            transform = transform_pipeline(data_config)
        
            datasets = MultiLabelClassification(
                dataset_cfg=data_config,
                class_names=deg_types ,
                transforms=transform
            )

            data_list.append(datasets)
            
            if accelerator.is_main_process:
                print(f" - Successfully loaded {data_name}: {len(datasets)} images")
        except Exception as error:
            if accelerator.is_main_process:
                print(f" [Warning] Failed to load dataset {data_name}: {error}")

    if not data_list:
        raise RuntimeError("No datasets were loaded. Please check your configuration paths.")

    full_dataset = ConcatDataset(data_list)
    
    if config.data.split_val:
        validation_size = int(len(full_dataset) * config.data.val_split)
        train_size = len(full_dataset) - validation_size
        
        train_dataset, validation_dataset = random_split(
            full_dataset, 
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
    else:
        train_dataset = full_dataset
        validation_dataset = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.dataloader.train.batch_size,
        num_workers=config.data.dataloader.train.num_workers,
        shuffle=config.data.dataloader.use_shuffle,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.data.dataloader.val.batch_size,
        num_workers=config.data.dataloader.val.num_workers,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    ) if validation_dataset else None


    ModelClass = MODEL_FACTORY[config.network.type]
    
    backbone_argument_name = "clip_type" if "CLIP" in config.network.type else "dino_type"
    
    model = ModelClass(
        feature_dim=config.network.feature_dim,
        num_types=config.network.num_classes,
        freeze_encoder=config.network.freeze_encoder,
        **{backbone_argument_name: config.network.backbone}
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.optim.lr,
        weight_decay=config.train.optim.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.scheduler.T_max,
        eta_min=config.train.scheduler.eta_min
    )
    
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.train.loss.loss_weight]).to(accelerator.device)
    )

    model, optimizer, train_dataloader, validation_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, scheduler
    )

    total_epochs = config.train.num_epochs
    global_step_counter = 0
    best_validation_accuracy = 0.0
    
    checkpoint_dir = os.path.join(config.experiments_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Let's Train! !🚀
    for epoch in range(total_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader, 
            disable=not accelerator.is_main_process, 
            desc=f"Epoch {epoch+1}/{total_epochs}"
        )
        
        for step, (images, labels) in enumerate(progress_bar):
            with accelerator.accumulate(model):
                _, _, logits = model(images)
                
                loss = loss_function(logits, labels)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                if step % config.logging.log_interval == 0 and accelerator.is_main_process:
                    current_learning_rate = optimizer.param_groups[0]["lr"]
                    
                    # 3. 使用 accelerator.log，它会自动发给 swanlab
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": current_learning_rate,
                        "epoch": epoch
                    }, step=global_step_counter)
                    
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
                global_step_counter += 1
        
        scheduler.step()
        

        should_validate = validation_dataloader is not None and (epoch + 1) % config.val.val_freq == 0
        
        if should_validate:
            model.eval()
            total_validation_loss = 0.0
            total_correct_predictions = 0.0
            total_validation_samples = 0.0
            
            for images, labels in validation_dataloader:
                with torch.no_grad():
                    # Get probabilities for metrics, logits for loss
                    _, probs, logits = model(images)
                    
                    batch_loss = loss_function(logits, labels)
                    total_validation_loss += batch_loss.item()
                    
                    # Use probabilities > 0.5 for thresholding
                    predictions = (probs > 0.5).float()
                    
                    num_classes = config.network.num_classes
                    correct_matches = (predictions == labels).sum(dim=1)
                    batch_correct_count = (correct_matches == num_classes).sum()
                    
                    total_correct_predictions += batch_correct_count
                    total_validation_samples += labels.size(0)
            

            metrics_tensor = torch.tensor(
                [total_validation_loss, total_correct_predictions, total_validation_samples], 
                device=accelerator.device
            )
            
            # [num_processes, 3]
            gathered_metrics = accelerator.gather(metrics_tensor)
            
            # gather all gpus
            global_val_loss_sum = gathered_metrics[:, 0].sum().item()
            global_correct_sum = gathered_metrics[:, 1].sum().item()
            global_samples_sum = gathered_metrics[:, 2].sum().item()
            
            # Calculate global averages
            total_batches_across_gpus = len(validation_dataloader) * accelerator.num_processes
            final_validation_loss = global_val_loss_sum / total_batches_across_gpus
            
            final_validation_accuracy = global_correct_sum / global_samples_sum if global_samples_sum > 0 else 0.0
            
            # Main process logging
            if accelerator.is_main_process:
                print(f"\nValidation Result - Epoch {epoch+1}:")
                print(f"  Loss: {final_validation_loss:.4f}")
                print(f"  Exact Accuracy: {final_validation_accuracy:.4f}")
                
                # 4. 同样使用 accelerator.log
                accelerator.log({
                    "val/loss": final_validation_loss,
                    "val/accuracy": final_validation_accuracy
                }, step=global_step_counter)
                
                unwrapped_model = accelerator.unwrap_model(model)
                
                # 1. Periodic Save
                if (epoch + 1) % config.train.model_save_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
                    torch.save(unwrapped_model.state_dict(), checkpoint_path)
                    print(f"  Checkpoint saved: {checkpoint_path}")
                
                # 2. Best Model Save
                if config.train.weight_save_best:
                    if final_validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = final_validation_accuracy
                        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                        torch.save(unwrapped_model.state_dict(), best_model_path)
                        print(f"  🏆 New best model saved! (Acc: {best_validation_accuracy:.4f})")

    # 5. 自动结束所有 tracker
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accelerate Training Script for Multi-Label Classification")
    parser.add_argument("--config", '-c',type=str, default="config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    train(args)
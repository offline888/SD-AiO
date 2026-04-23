import argparse
import ast


def parse_timestep(value):
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid list format: {value}")
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Flux.2 single-step image restoration fine-tuning."
    )

    # Model
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, required=True
    )
    parser.add_argument("--datasets_config", type=str, default=None, required=True)
    parser.add_argument("--resolution", type=int, default=512)

    # Training
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--val_monitor_steps",
        type=int,
        default=None,
        help="Set to 0 to disable in-training visual monitoring.",
    )
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--fixed_timestep", type=parse_timestep, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", choices=["AdamW", "prodigy"]
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # Degradation Classifier
    parser.add_argument("--degradation_classifier_path", type=str, default=None, required=True)
    parser.add_argument("--dino_type", type=str, default=None)
    parser.add_argument("--num_deg_types", type=int, default=4)

    # Modulation backbone type
    parser.add_argument(
        "--mod_lq_type",
        type=str,
        default="convnext",
        choices=["convnext", "vae"],
        help="Backbone type for FLUX2ModulationV2: 'convnext' (ConvNeXt-Small) or 'vae' (dedicated VAE encoder)",
    )

    # Accelerate config
    parser.add_argument("--output_dir", type=str, default="flux2-image-restoration")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="swanlab")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args(input_args)
    import os
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

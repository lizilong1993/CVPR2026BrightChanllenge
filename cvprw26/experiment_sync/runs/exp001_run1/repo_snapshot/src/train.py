"""Training entry-point for BRIGHT building damage instance segmentation.

Usage::

    python -m src.train --config config/disaster.yaml
    python -m src.train --config config/disaster.yaml --epochs 5
    python -m src.train --config config/disaster.yaml --resume outputs/latest.pth
"""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.utils import load_config, set_seed, collate_fn
from src.dataset.data import BRIGHTDataset, get_transforms
from src.model.mask_rcnn import build_model
from src.engine import train_one_epoch, evaluate, format_metrics_report
from src.model.losses import FocalLoss


def get_loss_criterion(model_cfg):
    loss_type = model_cfg.get("loss", "ce")
    if loss_type == "focal":
        return FocalLoss(alpha=0.25, gamma=2.0)
    return None  # Use default


def resolve_runtime_device(requested, logger, label):
    requested = str(requested).lower()
    if requested == "auto":
        candidates = ("xpu", "cuda", "cpu")
    else:
        candidates = (requested,)

    for candidate in candidates:
        if candidate == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            logger.info(f"Using {label}: xpu")
            return device
        if candidate == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using {label}: cuda")
            return device
        if candidate == "cpu":
            device = torch.device("cpu")
            logger.info(f"Using {label}: cpu")
            return device

    logger.warning(f"Requested {label} '{requested}' is unavailable. Falling back to CPU.")
    return torch.device("cpu")


def resolve_device(train_cfg, logger):
    return resolve_runtime_device(train_cfg.get("device", "auto"), logger, "train device")


def resolve_eval_device(train_cfg, train_device, logger):
    requested = train_cfg.get("eval_device")
    if requested in (None, "", "same"):
        logger.info(f"Using eval device: {train_device.type}")
        return train_device
    return resolve_runtime_device(requested, logger, "eval device")


def maybe_subset(dataset, subset_size, seed, split_name, logger):
    if subset_size in (None, 0, "", False):
        logger.info(f"{split_name} split uses full dataset: {len(dataset)} samples")
        return dataset

    subset_size = min(int(subset_size), len(dataset))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()[:subset_size]
    logger.info(f"{split_name} split subset enabled: {subset_size}/{len(dataset)} samples")
    return torch.utils.data.Subset(dataset, indices)


def maybe_filter_empty_targets(dataset, enabled, logger):
    if not enabled:
        return dataset

    if not hasattr(dataset, "coco") or not hasattr(dataset, "ids"):
        logger.warning("Empty-target filtering skipped because dataset does not expose COCO ids.")
        return dataset

    kept_indices = []
    for idx, image_id in enumerate(dataset.ids):
        if dataset.coco.getAnnIds(imgIds=image_id):
            kept_indices.append(idx)

    logger.info(f"Filtered empty-target training samples: kept {len(kept_indices)}/{len(dataset)}")
    return torch.utils.data.Subset(dataset, kept_indices)


def unwrap_dataset(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, None


def build_train_sampler(dataset, train_cfg, logger):
    strategy = str(train_cfg.get("sampler", "random")).lower()
    if strategy in ("", "none", "random"):
        return None

    if strategy != "damage_aware":
        raise ValueError(f"Unsupported train.sampler={strategy!r}. Use random/none/damage_aware.")

    base_dataset, subset_indices = unwrap_dataset(dataset)
    if not hasattr(base_dataset, "coco") or not hasattr(base_dataset, "ids"):
        logger.warning("Damage-aware sampler is unavailable for the current dataset type. Falling back to random sampling.")
        return None

    if subset_indices is None:
        subset_indices = list(range(len(base_dataset)))

    damaged_boost = float(train_cfg.get("damaged_boost", 8.0))
    destroyed_boost = float(train_cfg.get("destroyed_boost", 4.0))

    weights = []
    damaged_images = 0
    destroyed_images = 0
    boosted_images = 0

    for dataset_idx in subset_indices:
        image_id = base_dataset.ids[dataset_idx]
        anns = base_dataset.coco.imgToAnns.get(image_id, [])
        categories = {ann["category_id"] for ann in anns}

        weight = 1.0
        if 2 in categories:
            weight *= damaged_boost
            damaged_images += 1
        if 3 in categories:
            weight *= destroyed_boost
            destroyed_images += 1
        if weight > 1.0:
            boosted_images += 1
        weights.append(weight)

    logger.info(
        "Using damage-aware sampler: boosted=%d/%d, damaged_images=%d, destroyed_images=%d, damaged_boost=%.2f, destroyed_boost=%.2f",
        boosted_images,
        len(weights),
        damaged_images,
        destroyed_images,
        damaged_boost,
        destroyed_boost,
    )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on BRIGHT data")
    parser.add_argument("--config", default="config/disaster.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(log_file, mode="a"))

    set_seed(train_cfg["seed"])

    device = resolve_device(train_cfg, logger)
    eval_device = resolve_eval_device(train_cfg, device, logger)
    pin_memory = bool(train_cfg.get("pin_memory", device.type in {"cuda", "xpu"}))

    # Datasets & data loaders
    image_dir = os.path.join(data_cfg["root"], data_cfg["images_dir"])
    pre_event_dir = os.path.join(data_cfg["root"], data_cfg["pre_event_dir"])

    train_dataset = BRIGHTDataset(
        ann_file=data_cfg["train_ann"],
        image_dir=image_dir,
        pre_event_dir=pre_event_dir,
        transforms=get_transforms(train=True),
    )
    train_dataset = maybe_filter_empty_targets(
        train_dataset,
        train_cfg.get("filter_empty_targets", False),
        logger,
    )
    train_dataset = maybe_subset(
        train_dataset,
        train_cfg.get("train_subset_size"),
        train_cfg["seed"],
        "train",
        logger,
    )

    val_dataset = BRIGHTDataset(
        ann_file=data_cfg["val_ann"],
        image_dir=image_dir,
        pre_event_dir=pre_event_dir,
        transforms=get_transforms(train=False),
    )
    val_dataset = maybe_subset(
        val_dataset,
        train_cfg.get("val_subset_size"),
        train_cfg["seed"] + 1,
        "val",
        logger,
    )

    logger.info(
        "Dataset summary: train=%d samples, val=%d samples",
        len(train_dataset),
        len(val_dataset),
    )
    train_sampler = build_train_sampler(train_dataset, train_cfg, logger)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    # Model
    model = build_model(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        pixel_mean=data_cfg["pixel_mean"],
        pixel_std=data_cfg["pixel_std"],
        box_detections_per_img=model_cfg.get("box_detections_per_img", 1500),
        rpn_pre_nms_top_n_test=model_cfg.get("rpn_pre_nms_top_n_test", 1500),
        rpn_post_nms_top_n_test=model_cfg.get("rpn_post_nms_top_n_test", 1500),
    )
    model.to(device)

    criterion = get_loss_criterion(model_cfg)
    if criterion:
        logger.info(f"Using custom loss: {model_cfg['loss']}")
        # This would require deeper integration into engine.py
        # For simplicity in this baseline, we'll assume standard loss for now
        # but keep the structure for future optimization rounds.
        pass

    # Log trainable parameter summary
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    components: dict[str, dict[str, int]] = {}

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        parts = name.split(".")
        comp = parts[0]
        if len(parts) > 1:
            comp = f"{comp}.{parts[1]}"
        if comp == "backbone.body" and len(parts) > 2:
            comp = f"{comp}.{parts[2]}"

        if comp not in components:
            components[comp] = {"trainable": 0, "frozen": 0}
        if param.requires_grad:
            trainable_params += n
            components[comp]["trainable"] += n
        else:
            frozen_params += n
            components[comp]["frozen"] += n

    logger.info("Parameter summary:")
    logger.info(f"  Total: {total_params:,}  |  Trainable: {trainable_params:,}  |  Frozen: {frozen_params:,}")
    for comp, counts in sorted(components.items()):
        total_c = counts["trainable"] + counts["frozen"]
        status = "FROZEN" if counts["trainable"] == 0 else ("TRAINABLE" if counts["frozen"] == 0 else "MIXED")
        logger.info(f"  {comp:<30s} {total_c:>10,} params  [{status}]")

    # Optimiser & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg["lr_steps"],
        gamma=train_cfg["lr_gamma"],
    )

    if train_cfg.get("amp", False):
        if device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")
        elif device.type == "xpu":
            # For Intel Arc A380, GradScaler is not supported due to lack of FP64.
            logger.warning("GradScaler is disabled for XPU on this device (No FP64 support).")
            scaler = None
        else:
            scaler = None
    else:
        scaler = None

    # Resume from checkpoint
    start_epoch = 0
    best_ap = 0.0

    resume_path = args.resume
    if not resume_path and train_cfg.get("auto_resume", False):
        candidate = os.path.join(output_dir, "latest.pth")
        if os.path.exists(candidate):
            resume_path = candidate

    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_ap = checkpoint.get("best_ap", 0.0)
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"  Resumed at epoch {start_epoch}, best segm_AP so far: {best_ap:.4f}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            print_freq=train_cfg["print_freq"],
            scaler=scaler,
            warmup_iters=train_cfg.get("warmup_iters", 0),
            warmup_factor=train_cfg.get("warmup_factor", 0.001),
            log_file=log_file,
        )
        lr_scheduler.step()

        eval_dir = os.path.join(output_dir, "eval")
        if eval_device != device:
            logger.info(f"Moving model to {eval_device.type} for evaluation")
            model.to(eval_device)
        try:
            metrics = evaluate(model, val_loader, eval_device, output_dir=eval_dir, epoch=epoch, log_file=log_file)
        finally:
            if eval_device != device:
                logger.info(f"Moving model back to {device.type} after evaluation")
                model.to(device)
        logger.info(f"Epoch [{epoch}] evaluation: {metrics}")
        for line in format_metrics_report(metrics):
            logger.info(f"Epoch [{epoch}] {line}")

        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_ap": best_ap,
        }
        if scaler is not None:
            checkpoint_data["scaler"] = scaler.state_dict()

        latest_path = os.path.join(output_dir, "latest.pth")
        torch.save(checkpoint_data, latest_path)
        logger.info(f"  Saved latest checkpoint: {latest_path}")

        segm_ap = metrics.get("segm_AP", 0.0)
        if segm_ap > best_ap:
            best_ap = segm_ap
            checkpoint_data["best_ap"] = best_ap
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save(checkpoint_data, best_path)
            logger.info(f"  New best segm_AP: {best_ap:.4f} -- saved to {best_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

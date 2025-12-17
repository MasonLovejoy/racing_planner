import torch
import torch.nn as nn

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.utils.tensorboard as tb

from .models import load_model, save_model, PROJECT_DIR
from .datasets.road_dataset import load_data

DATA_ROOT = PROJECT_DIR.parent / "drive_data"


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 2,
    **kwargs,
):
    """
    Training pipeline for autonomous racing planners.
    
    Args:
        exp_dir (str): directory for experiment logs
        model_name (str): name of model architecture to train
        num_epoch (int): number of training epochs
        lr (float): learning rate
        batch_size (int): batch size for training
        seed (int): random seed for reproducibility
        num_workers (int): number of data loading workers
        **kwargs: additional model configuration parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Initialize model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Configure data pipelines based on model type
    if model_name == "cnn_planner":
        train_pipeline = "aug" 
        val_pipeline = "default"
    else:
        train_pipeline = "aug"
        val_pipeline = "state_only"

    # Load datasets
    train_loader = load_data(
        DATA_ROOT / "train", 
        transform_pipeline=train_pipeline, 
        shuffle=True, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    val_loader = load_data(
        DATA_ROOT / "val", 
        transform_pipeline=val_pipeline, 
        shuffle=False, 
        batch_size=batch_size, 
        num_workers=num_workers
    )

    # Setup training components
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epoch, 
        eta_min=1e-5
    )

    global_step = 0
    metrics = {
        "train_loss": [], 
        "val_loss": [], 
        "val_longitudinal_error": [], 
        "val_lateral_error": []
    }

    # Training loop
    for epoch in range(num_epoch):
        # Reset metrics
        for key in metrics:
            metrics[key].clear()

        model.train()

        # Training phase
        for batch in train_loader:
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass based on model type
            if model_name == "cnn_planner":
                image = batch['image'].to(device)
                waypoints_pred = model(image=image)
            else:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints_pred = model(track_left=track_left, track_right=track_right)

            # Compute masked loss
            mask_expanded = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)
            loss = loss_func(waypoints_pred[mask_expanded], waypoints[mask_expanded])
            metrics["train_loss"].append(loss.detach())

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # Validation phase
        with torch.inference_mode():
            model.eval()

            for batch in val_loader:
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                # Forward pass
                if model_name == "cnn_planner":
                    image = batch['image'].to(device)
                    waypoints_pred = model(image=image)
                else:
                    track_left = batch['track_left'].to(device)
                    track_right = batch['track_right'].to(device)
                    waypoints_pred = model(track_left=track_left, track_right=track_right)

                # Compute validation loss
                mask_expanded = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)
                loss = loss_func(waypoints_pred[mask_expanded], waypoints[mask_expanded])
                metrics["val_loss"].append(loss.detach())

                # Compute directional errors
                diff = waypoints_pred - waypoints
                valid_diff = diff[waypoints_mask]

                if valid_diff.numel() > 0:
                    longitudinal_error = torch.abs(valid_diff[:, 0]).mean()
                    lateral_error = torch.abs(valid_diff[:, 1]).mean()

                    metrics["val_longitudinal_error"].append(longitudinal_error.detach())
                    metrics["val_lateral_error"].append(lateral_error.detach())

        # Log metrics
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()
        logger.add_scalar("train/loss", epoch_train_loss, global_step=global_step)
        logger.add_scalar("val/loss", epoch_val_loss, global_step=global_step)

        epoch_val_long = torch.as_tensor(metrics["val_longitudinal_error"]).mean()
        epoch_val_lat = torch.as_tensor(metrics["val_lateral_error"]).mean()
        logger.add_scalar("val/longitudinal_error", epoch_val_long, global_step=global_step)
        logger.add_scalar("val/lateral_error", epoch_val_lat, global_step=global_step)

        # Print progress
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"long_err={epoch_val_long:.4f} "
                f"lat_err={epoch_val_lat:.4f}"
            )

    # Save final model
    save_model(model)

    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
    
    return model
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autonomous racing planner models")
    parser.add_argument("--exp_dir", type=str, default="logs", help="Experiment directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model architecture name")
    parser.add_argument("--num_epoch", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    
    train(**vars(parser.parse_args()))

"""
Example usage:
    python3 -m racing_planner.train_planner --model_name mlp_planner --num_epoch 50
    python3 -m racing_planner.train_planner --model_name transformer_planner --lr 1e-4
    python3 -m racing_planner.train_planner --model_name cnn_planner --batch_size 128
"""
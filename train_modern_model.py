#!/usr/bin/env python3
"""
Training script for the modern transformer-based pose forecasting model.
This script demonstrates how to train the state-of-the-art model and evaluate it.
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
import os
import sys
from typing import Optional

# Add src to path
sys.path.append('src')

from models.modern_pose_forecaster import ModernPoseForecaster
from utils.metrics import MPJPE, PDJ


class SyntheticPoseDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset for demonstration purposes.
    In a real scenario, you would use actual pose data from datasets like:
    - Human3.6M
    - CMU Motion Capture
    - PoseTrack
    """
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 20, num_joints: int = 17):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_joints = num_joints
        self.input_dim = num_joints * 2  # x, y coordinates
        
        # Generate synthetic pose sequences
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> torch.Tensor:
        """Generate synthetic pose sequences with realistic motion patterns"""
        data = []
        
        for _ in range(self.num_samples):
            # Generate base pose (standing position)
            base_pose = torch.randn(self.num_joints, 2) * 0.1
            
            # Add some structure to the pose
            # Head (joints 0-4)
            base_pose[0] = torch.tensor([0.5, 0.2])  # Nose
            base_pose[1] = torch.tensor([0.45, 0.15])  # Left eye
            base_pose[2] = torch.tensor([0.55, 0.15])  # Right eye
            base_pose[3] = torch.tensor([0.4, 0.1])   # Left ear
            base_pose[4] = torch.tensor([0.6, 0.1])   # Right ear
            
            # Shoulders (joints 5-6)
            base_pose[5] = torch.tensor([0.4, 0.3])   # Left shoulder
            base_pose[6] = torch.tensor([0.6, 0.3])   # Right shoulder
            
            # Arms (joints 7-10)
            base_pose[7] = torch.tensor([0.3, 0.4])   # Left elbow
            base_pose[8] = torch.tensor([0.7, 0.4])   # Right elbow
            base_pose[9] = torch.tensor([0.2, 0.5])   # Left wrist
            base_pose[10] = torch.tensor([0.8, 0.5])  # Right wrist
            
            # Hips (joints 11-12)
            base_pose[11] = torch.tensor([0.45, 0.6])  # Left hip
            base_pose[12] = torch.tensor([0.55, 0.6])  # Right hip
            
            # Legs (joints 13-16)
            base_pose[13] = torch.tensor([0.45, 0.8])  # Left knee
            base_pose[14] = torch.tensor([0.55, 0.8])  # Right knee
            base_pose[15] = torch.tensor([0.45, 1.0])  # Left ankle
            base_pose[16] = torch.tensor([0.55, 1.0])  # Right ankle
            
            # Generate sequence with motion
            sequence = []
            current_pose = base_pose.clone()
            
            for frame in range(self.seq_length):
                # Add some motion (walking, waving, etc.)
                motion = torch.randn_like(current_pose) * 0.02
                
                # Add periodic motion for arms (waving)
                if frame > 0:
                    wave_freq = 0.3
                    wave_amplitude = 0.05
                    wave = torch.sin(frame * wave_freq) * wave_amplitude
                    motion[9, 0] += wave  # Left wrist x
                    motion[10, 0] -= wave  # Right wrist x
                
                # Add walking motion for legs
                if frame > 0:
                    walk_freq = 0.5
                    walk_amplitude = 0.03
                    walk = torch.sin(frame * walk_freq) * walk_amplitude
                    motion[15, 0] += walk  # Left ankle x
                    motion[16, 0] -= walk  # Right ankle x
                
                current_pose = current_pose + motion
                sequence.append(current_pose.flatten())
            
            data.append(torch.stack(sequence))
        
        return torch.stack(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class PoseDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for pose forecasting"""
    
    def __init__(self, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, and testing"""
        if stage == 'fit' or stage is None:
            # Training and validation datasets
            train_size = int(0.8 * 1000)  # 80% for training
            val_size = 1000 - train_size
            
            full_dataset = SyntheticPoseDataset(num_samples=1000)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = SyntheticPoseDataset(num_samples=200)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def train_model(args):
    """Train the modern pose forecasting model"""
    
    # Set random seeds for reproducibility
    pl.seed_everything(args.seed)
    
    # Create data module
    data_module = PoseDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = ModernPoseForecaster(
        input_dim=34,  # 17 joints * 2 coordinates
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.learning_rate
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_total_loss',
            dirpath=args.checkpoint_dir,
            filename='pose_forecaster-{epoch:02d}-{val_total_loss:.4f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_total_loss',
            patience=args.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create loggers
    loggers = []
    
    if args.use_tensorboard:
        tensorboard_logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name='pose_forecasting'
        )
        loggers.append(tensorboard_logger)
    
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project='pose-forecasting',
            name=args.experiment_name
        )
        loggers.append(wandb_logger)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate every 25% of training epoch
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    print("Testing model...")
    test_results = trainer.test(model, data_module)
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
    print(f"Test MSE Loss: {test_results[0]['test_mse_loss']:.4f}")
    print(f"Test MAE Loss: {test_results[0]['test_mae_loss']:.4f}")
    print(f"Test MPJPE Error: {test_results[0]['test_mpjpe']:.4f}")
    print(f"Test PDJ Score: {test_results[0]['test_pdj_score']:.4f}")
    
    return model, trainer


def evaluate_model(model_path: str, data_module: PoseDataModule):
    """Evaluate a trained model"""
    
    # Load the model
    model = ModernPoseForecaster.load_from_checkpoint(model_path)
    model.eval()
    
    # Create trainer for evaluation
    trainer = pl.Trainer(accelerator='auto', devices=1)
    
    # Test the model
    results = trainer.test(model, data_module)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Modern Pose Forecasting Model')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    
    # System settings
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator (auto, cpu, gpu, tpu)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--experiment_name', type=str, default='pose_forecasting', help='Experiment name')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use TensorBoard logging')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    # Evaluation
    parser.add_argument('--evaluate', type=str, help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.evaluate:
        # Evaluate existing model
        data_module = PoseDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
        data_module.setup()
        evaluate_model(args.evaluate, data_module)
    else:
        # Train new model
        model, trainer = train_model(args)
        
        # Save the final model
        final_model_path = os.path.join(args.checkpoint_dir, 'final_model.ckpt')
        trainer.save_checkpoint(final_model_path)
        print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Comprehensive evaluation script to compare legacy models with modern transformer-based approach.
This script evaluates all models on the same dataset and provides detailed performance analysis.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import os
import sys
import time
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from models.modern_pose_forecaster import ModernPoseForecaster
from models.skeleton_model import SkeletonModel
from models.heatmap_model import HeatmapModel
from utils.metrics import MPJPE, PDJ


class ModelEvaluator:
    """Comprehensive model evaluator for pose forecasting"""
    
    def __init__(self, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        self.models = {}
        self.results = {}
        
    def load_legacy_models(self):
        """Load legacy models for comparison"""
        print("Loading legacy models...")
        
        # Skeleton-based models
        self.models['skeleton_gru'] = SkeletonModel(
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            lr=3e-3
        ).to(self.device)
        
        self.models['skeleton_lstm'] = SkeletonModel(
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            lr=3e-3
        ).to(self.device)
        
        # Heatmap-based models
        self.models['heatmap_gru'] = HeatmapModel(
            hidden_channels=128,
            num_layers=1,
            dropout=0.0,
            lr=3e-3
        ).to(self.device)
        
        self.models['heatmap_lstm'] = HeatmapModel(
            hidden_channels=128,
            num_layers=1,
            dropout=0.0,
            lr=3e-3
        ).to(self.device)
        
        print(f"Loaded {len(self.models)} legacy models")
    
    def load_modern_model(self, model_path: str = None):
        """Load modern transformer-based model"""
        print("Loading modern transformer model...")
        
        if model_path and os.path.exists(model_path):
            self.models['modern_transformer'] = ModernPoseForecaster.load_from_checkpoint(model_path)
        else:
            self.models['modern_transformer'] = ModernPoseForecaster(
                input_dim=34,
                d_model=256,
                num_heads=8,
                num_layers=6,
                d_ff=1024,
                dropout=0.1,
                lr=1e-4
            )
        
        self.models['modern_transformer'] = self.models['modern_transformer'].to(self.device)
        print("Modern transformer model loaded")
    
    def generate_test_data(self, num_sequences: int = 100, seq_length: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate test data for evaluation"""
        print(f"Generating {num_sequences} test sequences...")
        
        # Generate realistic pose sequences
        test_data = []
        
        for _ in range(num_sequences):
            # Generate base pose
            base_pose = torch.randn(17, 2) * 0.1
            
            # Add structure to pose
            base_pose[0] = torch.tensor([0.5, 0.2])   # Nose
            base_pose[1] = torch.tensor([0.45, 0.15]) # Left eye
            base_pose[2] = torch.tensor([0.55, 0.15]) # Right eye
            base_pose[5] = torch.tensor([0.4, 0.3])   # Left shoulder
            base_pose[6] = torch.tensor([0.6, 0.3])   # Right shoulder
            base_pose[11] = torch.tensor([0.45, 0.6]) # Left hip
            base_pose[12] = torch.tensor([0.55, 0.6]) # Right hip
            
            sequence = []
            current_pose = base_pose.clone()
            
            for frame in range(seq_length):
                # Add motion
                motion = torch.randn_like(current_pose) * 0.02
                
                # Add periodic motion
                if frame > 0:
                    wave = torch.sin(frame * 0.3) * 0.05
                    motion[9, 0] += wave   # Left wrist
                    motion[10, 0] -= wave  # Right wrist
                    
                    walk = torch.sin(frame * 0.5) * 0.03
                    motion[15, 0] += walk  # Left ankle
                    motion[16, 0] -= walk  # Right ankle
                
                current_pose = current_pose + motion
                sequence.append(current_pose.flatten())
            
            test_data.append(torch.stack(sequence))
        
        test_data = torch.stack(test_data)
        
        # Split into input and target
        input_sequences = test_data[:, :10, :]  # First 10 frames
        target_sequences = test_data[:, 10:, :]  # Last 10 frames
        
        return input_sequences.to(self.device), target_sequences.to(self.device)
    
    def evaluate_model(self, model_name: str, model: torch.nn.Module, 
                      input_sequences: torch.Tensor, target_sequences: torch.Tensor) -> Dict:
        """Evaluate a single model"""
        print(f"Evaluating {model_name}...")
        
        model.eval()
        predictions = []
        inference_times = []
        
        with torch.no_grad():
            for i in tqdm(range(len(input_sequences)), desc=f"Evaluating {model_name}"):
                input_seq = input_sequences[i:i+1]  # Add batch dimension
                
                # Measure inference time
                start_time = time.time()
                
                if model_name == 'modern_transformer':
                    pred = model.predict_sequence(input_seq, num_frames=10)
                else:
                    pred = model.predict(input_seq, samples=10)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                predictions.append(pred.squeeze())
        
        predictions = torch.stack(predictions)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, target_sequences)
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['std_inference_time'] = np.std(inference_times)
        metrics['total_inference_time'] = np.sum(inference_times)
        
        return metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        # Basic losses
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)
        mae_loss = torch.nn.functional.l1_loss(predictions, targets)
        
        # Pose-specific metrics
        mpjpe_error = MPJPE(predictions, targets)
        pdj_score = PDJ(predictions, targets)
        
        # Additional metrics
        # Velocity error (how well the model predicts motion)
        pred_velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
        target_velocity = targets[:, 1:, :] - targets[:, :-1, :]
        velocity_error = torch.nn.functional.mse_loss(pred_velocity, target_velocity)
        
        # Acceleration error
        pred_acceleration = pred_velocity[:, 1:, :] - pred_velocity[:, :-1, :]
        target_acceleration = target_velocity[:, 1:, :] - target_velocity[:, :-1, :]
        acceleration_error = torch.nn.functional.mse_loss(pred_acceleration, target_acceleration)
        
        # Per-joint analysis
        joint_errors = []
        for joint_idx in range(17):
            joint_pred = predictions.view(-1, 17, 2)[:, joint_idx, :]
            joint_target = targets.view(-1, 17, 2)[:, joint_idx, :]
            joint_error = torch.norm(joint_pred - joint_target, dim=-1).mean()
            joint_errors.append(joint_error.item())
        
        return {
            'MSE': mse_loss.item(),
            'MAE': mae_loss.item(),
            'MPJPE': mpjpe_error.item(),
            'PDJ': pdj_score.item(),
            'Velocity_Error': velocity_error.item(),
            'Acceleration_Error': acceleration_error.item(),
            'Joint_Errors': joint_errors
        }
    
    def run_comprehensive_evaluation(self, num_sequences: int = 100) -> pd.DataFrame:
        """Run comprehensive evaluation of all models"""
        print("Starting comprehensive evaluation...")
        
        # Generate test data
        input_sequences, target_sequences = self.generate_test_data(num_sequences)
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}")
            print(f"{'='*50}")
            
            try:
                metrics = self.evaluate_model(model_name, model, input_sequences, target_sequences)
                self.results[model_name] = metrics
                print(f"✅ {model_name} evaluation completed")
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Create results DataFrame
        results_df = self.create_results_dataframe()
        
        return results_df
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Create a comprehensive results DataFrame"""
        # Extract main metrics
        main_metrics = ['MSE', 'MAE', 'MPJPE', 'PDJ', 'Velocity_Error', 'Acceleration_Error', 
                       'avg_inference_time', 'std_inference_time', 'total_inference_time']
        
        results_data = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                row = {'Model': model_name}
                for metric in main_metrics:
                    row[metric] = metrics.get(metric, np.nan)
                results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # Add rankings
        for metric in ['MSE', 'MAE', 'MPJPE', 'Velocity_Error', 'Acceleration_Error', 'avg_inference_time']:
            if metric in df.columns:
                df[f'{metric}_Rank'] = df[metric].rank(ascending=metric != 'PDJ')
        
        return df
    
    def create_visualizations(self, results_df: pd.DataFrame, save_dir: str = 'evaluation_results'):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall performance comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['MSE', 'MAE', 'MPJPE', 'PDJ', 'Velocity_Error', 'avg_inference_time']
        metric_names = ['MSE Loss', 'MAE Loss', 'MPJPE Error', 'PDJ Score', 'Velocity Error', 'Inference Time (s)']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 3, i % 3]
            
            if metric in results_df.columns:
                bars = ax.bar(results_df['Model'], results_df[metric])
                ax.set_title(name)
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart for overall performance
        self.create_radar_chart(results_df, save_dir)
        
        # 3. Per-joint error analysis
        self.create_joint_analysis_plot(save_dir)
        
        # 4. Inference time analysis
        self.create_inference_analysis(results_df, save_dir)
        
        print(f"Visualizations saved to {save_dir}")
    
    def create_radar_chart(self, results_df: pd.DataFrame, save_dir: str):
        """Create radar chart for overall performance"""
        # Normalize metrics for radar chart
        metrics_to_plot = ['MSE', 'MAE', 'MPJPE', 'Velocity_Error', 'Acceleration_Error']
        
        # Normalize (lower is better for all these metrics)
        normalized_data = results_df[metrics_to_plot].copy()
        for metric in metrics_to_plot:
            if metric in normalized_data.columns:
                min_val = normalized_data[metric].min()
                max_val = normalized_data[metric].max()
                normalized_data[metric] = 1 - (normalized_data[metric] - min_val) / (max_val - min_val)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, model_name in enumerate(results_df['Model']):
            values = normalized_data.iloc[idx].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot)
        ax.set_ylim(0, 1)
        ax.set_title('Normalized Performance Comparison (Higher is Better)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_joint_analysis_plot(self, save_dir: str):
        """Create per-joint error analysis"""
        joint_names = [
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
        ]
        
        # Get joint errors for modern model
        if 'modern_transformer' in self.results and 'Joint_Errors' in self.results['modern_transformer']:
            joint_errors = self.results['modern_transformer']['Joint_Errors']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(joint_names, joint_errors)
            ax.set_title('Per-Joint Prediction Error (Modern Transformer)')
            ax.set_ylabel('Average Error (pixels)')
            ax.tick_params(axis='x', rotation=45)
            
            # Color code by body part
            colors = ['red'] * 5 + ['blue'] * 6 + ['green'] * 6  # Head, Arms, Legs
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'joint_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_inference_analysis(self, results_df: pd.DataFrame, save_dir: str):
        """Create inference time analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Inference time comparison
        if 'avg_inference_time' in results_df.columns:
            bars = ax1.bar(results_df['Model'], results_df['avg_inference_time'])
            ax1.set_title('Average Inference Time')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add error bars
            if 'std_inference_time' in results_df.columns:
                ax1.errorbar(results_df['Model'], results_df['avg_inference_time'], 
                           yerr=results_df['std_inference_time'], fmt='none', color='black', capsize=5)
        
        # Speed vs Accuracy trade-off
        if 'MSE' in results_df.columns and 'avg_inference_time' in results_df.columns:
            scatter = ax2.scatter(results_df['avg_inference_time'], results_df['MSE'], 
                                s=100, alpha=0.7)
            ax2.set_xlabel('Average Inference Time (seconds)')
            ax2.set_ylabel('MSE Loss')
            ax2.set_title('Speed vs Accuracy Trade-off')
            
            # Add model labels
            for i, model_name in enumerate(results_df['Model']):
                ax2.annotate(model_name, (results_df['avg_inference_time'].iloc[i], results_df['MSE'].iloc[i]),
                           xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results_df: pd.DataFrame, save_dir: str = 'evaluation_results'):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save DataFrame
        results_df.to_csv(os.path.join(save_dir, 'evaluation_results.csv'), index=False)
        
        # Save detailed results
        with open(os.path.join(save_dir, 'detailed_results.txt'), 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, metrics in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                if 'error' in metrics:
                    f.write(f"Error: {metrics['error']}\n")
                else:
                    for key, value in metrics.items():
                        if key != 'Joint_Errors':
                            f.write(f"{key}: {value:.6f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                f.write("\n")
        
        print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    parser.add_argument('--modern_model_path', type=str, help='Path to modern model checkpoint')
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of test sequences')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Results save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Load models
    evaluator.load_legacy_models()
    evaluator.load_modern_model(args.modern_model_path)
    
    # Run evaluation
    results_df = evaluator.run_comprehensive_evaluation(args.num_sequences)
    
    # Create visualizations
    evaluator.create_visualizations(results_df, args.save_dir)
    
    # Save results
    evaluator.save_results(results_df, args.save_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Find best model for each metric
    print("\n" + "="*60)
    print("BEST MODELS BY METRIC")
    print("="*60)
    
    metrics = ['MSE', 'MAE', 'MPJPE', 'PDJ', 'avg_inference_time']
    metric_names = ['MSE Loss', 'MAE Loss', 'MPJPE Error', 'PDJ Score', 'Inference Time']
    
    for metric, name in zip(metrics, metric_names):
        if metric in results_df.columns:
            if metric == 'PDJ':
                best_model = results_df.loc[results_df[metric].idxmax(), 'Model']
                best_value = results_df[metric].max()
            else:
                best_model = results_df.loc[results_df[metric].idxmin(), 'Model']
                best_value = results_df[metric].min()
            
            print(f"{name}: {best_model} ({best_value:.6f})")


if __name__ == "__main__":
    main() 
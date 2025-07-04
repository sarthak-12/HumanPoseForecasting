#!/usr/bin/env python3
"""
Simple demo script for the Human Pose Forecasting System.
This script demonstrates the basic functionality without requiring all dependencies.
"""

import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append('src')

def demo_modern_model():
    """Demo the modern transformer model"""
    print("🤖 Modern Transformer Model Demo")
    print("=" * 40)
    
    try:
        from models.modern_pose_forecaster import ModernPoseForecaster
        import torch
        
        # Create a small model for demo
        model = ModernPoseForecaster(
            input_dim=34,      # 17 joints * 2 coordinates
            d_model=64,        # Small for demo
            num_heads=4,
            num_layers=2,
            d_ff=128,
            dropout=0.1
        )
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create demo input (simulating 10 frames of pose data)
        demo_input = torch.randn(1, 10, 34)  # 1 sequence, 10 frames, 34 features
        
        print(f"📊 Input shape: {demo_input.shape}")
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions = model.predict_sequence(demo_input, num_frames=5)
            inference_time = time.time() - start_time
        
        print(f"🔮 Predicted {predictions.shape[1]} future frames")
        print(f"⏱️  Inference time: {inference_time:.4f} seconds")
        print(f"📈 Output shape: {predictions.shape}")
        
        # Show some statistics
        print(f"📊 Prediction stats:")
        print(f"   - Mean: {predictions.mean():.4f}")
        print(f"   - Std: {predictions.std():.4f}")
        print(f"   - Min: {predictions.min():.4f}")
        print(f"   - Max: {predictions.max():.4f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping model demo: {e}")
        print("   Install PyTorch and dependencies to run this demo")
        return False
    except Exception as e:
        print(f"❌ Model demo failed: {e}")
        return False

def demo_pose_detection():
    """Demo the pose detection system"""
    print("\n🎯 Pose Detection Demo")
    print("=" * 40)
    
    try:
        from utils.pose_detector import MediaPipePoseDetector
        import cv2
        
        # Create detector
        detector = MediaPipePoseDetector(
            model_complexity=0,  # Fastest for demo
            min_detection_confidence=0.3
        )
        
        print("✅ MediaPipe pose detector initialized")
        
        # Create a demo image (simulating a person)
        demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a simple "stick figure" for demo
        # Head
        cv2.circle(demo_image, (320, 100), 30, (255, 255, 255), -1)
        # Body
        cv2.line(demo_image, (320, 130), (320, 300), (255, 255, 255), 3)
        # Arms
        cv2.line(demo_image, (320, 150), (250, 200), (255, 255, 255), 3)
        cv2.line(demo_image, (320, 150), (390, 200), (255, 255, 255), 3)
        # Legs
        cv2.line(demo_image, (320, 300), (280, 400), (255, 255, 255), 3)
        cv2.line(demo_image, (320, 300), (360, 400), (255, 255, 255), 3)
        
        print("📸 Created demo image with stick figure")
        
        # Try to detect pose
        pose = detector.detect_pose(demo_image)
        
        if pose is not None:
            print(f"✅ Pose detected with {pose.shape[0]} keypoints")
            
            # Show some keypoint positions
            keypoint_names = [
                'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
                'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
                'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
                'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
            ]
            
            print("📍 Sample keypoint positions:")
            for i, (name, pos) in enumerate(zip(keypoint_names[:5], pose[:5])):
                if pos[0] > 0 and pos[1] > 0:
                    print(f"   {name}: ({pos[0]:.1f}, {pos[1]:.1f})")
        else:
            print("⚠️  No pose detected in demo image")
            print("   (This is normal for a simple stick figure)")
        
        # Test pose sequence
        sequence = detector.get_pose_sequence(num_frames=5)
        if sequence is not None:
            print(f"📈 Pose sequence available: {sequence.shape}")
        else:
            print("📈 Pose sequence: Insufficient history (normal for demo)")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping pose detection demo: {e}")
        print("   Install MediaPipe and OpenCV to run this demo")
        return False
    except Exception as e:
        print(f"❌ Pose detection demo failed: {e}")
        return False

def demo_metrics():
    """Demo the evaluation metrics"""
    print("\n📊 Evaluation Metrics Demo")
    print("=" * 40)
    
    try:
        from utils.metrics import MPJPE, PDJ
        import torch
        
        # Create demo predictions and targets
        batch_size = 2
        num_frames = 5
        num_joints = 17
        
        # Generate realistic demo data
        predictions = torch.randn(batch_size, num_frames, num_joints * 2) * 0.1
        targets = predictions + torch.randn_like(predictions) * 0.05  # Close to predictions
        
        print(f"📊 Demo data created:")
        print(f"   - Predictions: {predictions.shape}")
        print(f"   - Targets: {targets.shape}")
        
        # Calculate metrics
        mpjpe_error = MPJPE(predictions, targets)
        pdj_score = PDJ(predictions, targets)
        
        print(f"📈 Metrics calculated:")
        print(f"   - MPJPE Error: {mpjpe_error:.6f}")
        print(f"   - PDJ Score: {pdj_score:.6f}")
        
        # Show what these metrics mean
        print(f"📝 Metric interpretation:")
        print(f"   - MPJPE: Lower is better (joint position accuracy)")
        print(f"   - PDJ: Higher is better (joint detection rate)")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping metrics demo: {e}")
        print("   Install PyTorch to run this demo")
        return False
    except Exception as e:
        print(f"❌ Metrics demo failed: {e}")
        return False

def demo_synthetic_data():
    """Demo synthetic data generation"""
    print("\n🎲 Synthetic Data Generation Demo")
    print("=" * 40)
    
    try:
        import torch
        
        # Generate realistic pose sequences
        num_samples = 5
        seq_length = 10
        num_joints = 17
        
        print(f"🎯 Generating {num_samples} pose sequences...")
        
        sequences = []
        for sample in range(num_samples):
            # Create base pose (standing position)
            base_pose = torch.randn(num_joints, 2) * 0.1
            
            # Add realistic structure
            base_pose[0] = torch.tensor([0.5, 0.2])   # Nose
            base_pose[1] = torch.tensor([0.45, 0.15]) # Left eye
            base_pose[2] = torch.tensor([0.55, 0.15]) # Right eye
            base_pose[5] = torch.tensor([0.4, 0.3])   # Left shoulder
            base_pose[6] = torch.tensor([0.6, 0.3])   # Right shoulder
            base_pose[11] = torch.tensor([0.45, 0.6]) # Left hip
            base_pose[12] = torch.tensor([0.55, 0.6]) # Right hip
            
            # Generate sequence with motion
            sequence = []
            current_pose = base_pose.clone()
            
            for frame in range(seq_length):
                # Add realistic motion
                motion = torch.randn_like(current_pose) * 0.02
                
                # Add periodic motion (like walking or waving)
                if frame > 0:
                    wave = torch.sin(torch.tensor(frame * 0.5)) * 0.03
                    motion[9, 0] += wave   # Left wrist
                    motion[10, 0] -= wave  # Right wrist
                    
                    walk = torch.sin(torch.tensor(frame * 0.3)) * 0.02
                    motion[15, 0] += walk  # Left ankle
                    motion[16, 0] -= walk  # Right ankle
                
                current_pose = current_pose + motion
                sequence.append(current_pose.flatten())
            
            sequences.append(torch.stack(sequence))
        
        sequences = torch.stack(sequences)
        
        print(f"✅ Generated pose sequences: {sequences.shape}")
        print(f"   - Samples: {sequences.shape[0]}")
        print(f"   - Frames: {sequences.shape[1]}")
        print(f"   - Features: {sequences.shape[2]} (17 joints × 2 coordinates)")
        
        # Show some statistics
        print(f"📊 Data statistics:")
        print(f"   - Mean: {sequences.mean():.4f}")
        print(f"   - Std: {sequences.std():.4f}")
        print(f"   - Range: [{sequences.min():.4f}, {sequences.max():.4f}]")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Skipping synthetic data demo: {e}")
        print("   Install PyTorch to run this demo")
        return False
    except Exception as e:
        print(f"❌ Synthetic data demo failed: {e}")
        return False

def demo_system_overview():
    """Demo the overall system architecture"""
    print("\n🏗️  System Architecture Overview")
    print("=" * 40)
    
    print("🎯 Components:")
    print("   1. MediaPipe Pose Detection")
    print("      - Real-time 33-landmark detection")
    print("      - COCO format conversion (17 keypoints)")
    print("      - Temporal smoothing")
    
    print("\n   2. Modern Transformer Model")
    print("      - Multi-head attention mechanism")
    print("      - Positional encoding")
    print("      - Autoregressive generation")
    print("      - Residual connections")
    
    print("\n   3. Evaluation Metrics")
    print("      - MSE/MAE for overall accuracy")
    print("      - MPJPE for joint-specific accuracy")
    print("      - PDJ for detection rate")
    print("      - Velocity/Acceleration errors")
    
    print("\n   4. Web Demonstrator")
    print("      - Real-time pose detection")
    print("      - Interactive parameter adjustment")
    print("      - Performance monitoring")
    print("      - Visualization tools")
    
    print("\n🚀 Key Improvements over Legacy:")
    print("   - Transformer architecture vs GRU/LSTM")
    print("   - MediaPipe vs basic pose estimation")
    print("   - Real-time capabilities")
    print("   - Comprehensive evaluation")
    print("   - Interactive web interface")

def main():
    """Run the demo"""
    print("🎪 Human Pose Forecasting System Demo")
    print("=" * 50)
    print("This demo shows the key components of the system.")
    print("Some demos may be skipped if dependencies are not installed.\n")
    
    demos = [
        ("System Overview", demo_system_overview),
        ("Synthetic Data", demo_synthetic_data),
        ("Modern Model", demo_modern_model),
        ("Pose Detection", demo_pose_detection),
        ("Evaluation Metrics", demo_metrics)
    ]
    
    successful_demos = 0
    total_demos = len(demos)
    
    for demo_name, demo_func in demos:
        try:
            if demo_func():
                successful_demos += 1
        except Exception as e:
            print(f"❌ {demo_name} demo error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Demo Results: {successful_demos}/{total_demos} demos successful")
    
    if successful_demos > 0:
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python test_system.py")
        print("3. Start web demo: streamlit run web_demonstrator.py")
        print("4. Train model: python train_modern_model.py")
    else:
        print("\n⚠️  No demos completed successfully.")
        print("Please install the required dependencies first.")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 
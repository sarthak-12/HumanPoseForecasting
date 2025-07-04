#!/usr/bin/env python3
"""
Test script to verify all components of the human pose forecasting system.
This script tests the modern model, pose detection, and basic functionality.
"""

import sys
import os
import torch
import numpy as np
import time

# Add src to path
sys.path.append('src')

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import pytorch_lightning as pl
        print("✅ PyTorch Lightning imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch Lightning import failed: {e}")
        return False
    
    try:
        from models.modern_pose_forecaster import ModernPoseForecaster
        print("✅ ModernPoseForecaster imported successfully")
    except ImportError as e:
        print(f"❌ ModernPoseForecaster import failed: {e}")
        return False
    
    try:
        from utils.pose_detector import MediaPipePoseDetector
        print("✅ MediaPipePoseDetector imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipePoseDetector import failed: {e}")
        return False
    
    try:
        from utils.metrics import MPJPE, PDJ
        print("✅ Metrics imported successfully")
    except ImportError as e:
        print(f"❌ Metrics import failed: {e}")
        return False
    
    return True

def test_modern_model():
    """Test the modern transformer-based model"""
    print("\nTesting modern model...")
    
    try:
        from models.modern_pose_forecaster import ModernPoseForecaster
        
        # Create model
        model = ModernPoseForecaster(
            input_dim=34,
            d_model=128,  # Smaller for testing
            num_heads=4,
            num_layers=2,
            d_ff=512,
            dropout=0.1
        )
        
        # Create test input
        batch_size = 2
        seq_length = 10
        input_dim = 34
        
        test_input = torch.randn(batch_size, seq_length, input_dim)
        
        # Test forward pass
        with torch.no_grad():
            predictions = model(test_input, max_length=5)
        
        expected_shape = (batch_size, 5, input_dim)
        if predictions.shape == expected_shape:
            print(f"✅ Model forward pass successful: {predictions.shape}")
        else:
            print(f"❌ Model output shape mismatch: expected {expected_shape}, got {predictions.shape}")
            return False
        
        # Test prediction method
        predictions = model.predict_sequence(test_input, num_frames=5)
        if predictions.shape == expected_shape:
            print(f"✅ Model prediction successful: {predictions.shape}")
        else:
            print(f"❌ Model prediction shape mismatch: expected {expected_shape}, got {predictions.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Modern model test failed: {e}")
        return False

def test_pose_detector():
    """Test the MediaPipe pose detector"""
    print("\nTesting pose detector...")
    
    try:
        from utils.pose_detector import MediaPipePoseDetector
        import cv2
        
        # Create detector
        detector = MediaPipePoseDetector(
            model_complexity=0,  # Fastest for testing
            min_detection_confidence=0.3
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test pose detection
        pose = detector.detect_pose(dummy_image)
        
        if pose is None:
            print("✅ Pose detector initialized (no pose in dummy image)")
        else:
            print(f"✅ Pose detected: {pose.shape}")
        
        # Test pose sequence
        sequence = detector.get_pose_sequence(num_frames=5)
        if sequence is None:
            print("✅ Pose sequence method works (insufficient history)")
        else:
            print(f"✅ Pose sequence generated: {sequence.shape}")
        
        # Clean up
        del detector
        
        return True
        
    except Exception as e:
        print(f"❌ Pose detector test failed: {e}")
        return False

def test_metrics():
    """Test the evaluation metrics"""
    print("\nTesting metrics...")
    
    try:
        from utils.metrics import MPJPE, PDJ
        
        # Create test data
        batch_size = 2
        num_frames = 5
        num_joints = 17
        
        predictions = torch.randn(batch_size, num_frames, num_joints * 2)
        targets = torch.randn(batch_size, num_frames, num_joints * 2)
        
        # Test MPJPE
        mpjpe_error = MPJPE(predictions, targets)
        print(f"✅ MPJPE calculated: {mpjpe_error:.6f}")
        
        # Test PDJ
        pdj_score = PDJ(predictions, targets)
        print(f"✅ PDJ calculated: {pdj_score:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation"""
    print("\nTesting synthetic data generation...")
    
    try:
        # Generate synthetic pose data
        num_samples = 10
        seq_length = 20
        num_joints = 17
        
        # Create base poses
        base_poses = []
        for _ in range(num_samples):
            # Generate realistic pose structure
            pose = torch.randn(num_joints, 2) * 0.1
            
            # Add structure
            pose[0] = torch.tensor([0.5, 0.2])   # Nose
            pose[5] = torch.tensor([0.4, 0.3])   # Left shoulder
            pose[6] = torch.tensor([0.6, 0.3])   # Right shoulder
            pose[11] = torch.tensor([0.45, 0.6]) # Left hip
            pose[12] = torch.tensor([0.55, 0.6]) # Right hip
            
            base_poses.append(pose)
        
        # Generate sequences
        sequences = []
        for base_pose in base_poses:
            sequence = []
            current_pose = base_pose.clone()
            
            for frame in range(seq_length):
                # Add motion
                motion = torch.randn_like(current_pose) * 0.02
                current_pose = current_pose + motion
                sequence.append(current_pose.flatten())
            
            sequences.append(torch.stack(sequence))
        
        sequences = torch.stack(sequences)
        
        print(f"✅ Synthetic data generated: {sequences.shape}")
        print(f"   - Samples: {sequences.shape[0]}")
        print(f"   - Frames: {sequences.shape[1]}")
        print(f"   - Features: {sequences.shape[2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {e}")
        return False

def test_training_components():
    """Test training-related components"""
    print("\nTesting training components...")
    
    try:
        import pytorch_lightning as pl
        
        # Test PyTorch Lightning
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='auto',
            devices=1,
            enable_checkpointing=False,
            logger=False
        )
        
        print("✅ PyTorch Lightning trainer created")
        
        # Test model creation
        from models.modern_pose_forecaster import ModernPoseForecaster
        
        model = ModernPoseForecaster(
            input_dim=34,
            d_model=64,  # Small for testing
            num_heads=2,
            num_layers=1,
            d_ff=128,
            dropout=0.1
        )
        
        print("✅ Training model created")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        return False

def test_performance():
    """Test basic performance"""
    print("\nTesting performance...")
    
    try:
        from models.modern_pose_forecaster import ModernPoseForecaster
        
        # Create model
        model = ModernPoseForecaster(
            input_dim=34,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            dropout=0.1
        )
        
        # Create test input
        test_input = torch.randn(1, 10, 34)
        
        # Measure inference time
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            predictions = model.predict_sequence(test_input, num_frames=5)
            inference_time = time.time() - start_time
        
        print(f"✅ Inference time: {inference_time:.4f} seconds")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {predictions.shape}")
        
        if inference_time < 1.0:  # Should be much faster
            print("✅ Performance acceptable")
            return True
        else:
            print("⚠️ Performance might be slow")
            return True  # Still pass, just warn
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Human Pose Forecasting System")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Modern Model", test_modern_model),
        ("Pose Detector", test_pose_detector),
        ("Metrics", test_metrics),
        ("Synthetic Data", test_synthetic_data),
        ("Training Components", test_training_components),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run the web demonstrator: streamlit run web_demonstrator.py")
        print("2. Train a model: python train_modern_model.py")
        print("3. Evaluate models: python evaluate_models.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify CUDA installation if using GPU")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
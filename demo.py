#!/usr/bin/env python3
"""
MeowTrix-AI Demo Script
Demonstrates the capabilities of the deepfake detection system
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_sample_images():
    """Create sample images for demonstration"""

    # Create directories
    os.makedirs('demo_data/real_faces', exist_ok=True)
    os.makedirs('demo_data/fake_faces', exist_ok=True)

    print("📸 Creating sample demonstration images...")

    # Generate sample "real" faces (random noise with face-like patterns)
    for i in range(10):
        # Create a face-like pattern
        img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

        # Add some structure to make it look more face-like
        center_y, center_x = 64, 64

        # Eyes
        cv2.circle(img, (45, 45), 8, (0, 0, 0), -1)
        cv2.circle(img, (83, 45), 8, (0, 0, 0), -1)

        # Nose
        cv2.line(img, (64, 60), (64, 80), (100, 100, 100), 2)

        # Mouth
        cv2.ellipse(img, (64, 95), (15, 8), 0, 0, 180, (50, 50, 50), 2)

        # Save image
        img_pil = Image.fromarray(img)
        img_pil.save(f'demo_data/real_faces/real_{i:03d}.jpg')

    # Generate sample "fake" faces (slightly different patterns)
    for i in range(10):
        # Create a different pattern for "fake" faces
        img = np.random.randint(30, 180, (128, 128, 3), dtype=np.uint8)

        # Add slightly different structure
        center_y, center_x = 64, 64

        # Different eye pattern
        cv2.rectangle(img, (40, 40), (50, 50), (0, 0, 0), -1)
        cv2.rectangle(img, (78, 40), (88, 50), (0, 0, 0), -1)

        # Different nose
        cv2.rectangle(img, (62, 60), (66, 80), (120, 120, 120), -1)

        # Different mouth
        cv2.rectangle(img, (50, 90), (78, 100), (70, 70, 70), -1)

        # Save image
        img_pil = Image.fromarray(img)
        img_pil.save(f'demo_data/fake_faces/fake_{i:03d}.jpg')

    print("✅ Sample images created in demo_data/ directory")
    return True

def run_training_demo():
    """Run a quick training demonstration"""
    print("\n🏋️ Starting MeowTrix-AI Training Demo...")

    try:
        # Import the training module
        from train_meowtrix import MeowTrixTrainer

        # Create trainer
        trainer = MeowTrixTrainer()

        # Run training with demo data
        results = trainer.run_complete_training(
            'demo_data/real_faces',
            'demo_data/fake_faces', 
            'demo_models/demo_model.joblib',
            models_to_train=['svm_linear']  # Quick training
        )

        print("✅ Demo training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Demo training failed: {str(e)}")
        return False

def run_detection_demo():
    """Run detection demonstration"""
    print("\n🔍 Starting Detection Demo...")

    try:
        from interface import MeowTrixCLI

        cli = MeowTrixCLI()

        # Test with a sample image
        if os.path.exists('demo_data/real_faces/real_000.jpg'):
            result = cli.detect_single_image(
                'demo_data/real_faces/real_000.jpg',
                'demo_models/demo_model.joblib'
            )

            print("🎯 Detection Result:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")

        return True

    except Exception as e:
        print(f"❌ Detection demo failed: {str(e)}")
        return False

def main():
    """Run complete demo"""
    print("🐱 Welcome to MeowTrix-AI Demo!")
    print("=" * 50)

    # Step 1: Create sample data
    if not create_sample_images():
        return

    # Step 2: Run training demo
    if not run_training_demo():
        print("⚠️ Training demo failed, but you can still try the GUI!")

    # Step 3: Run detection demo
    if not run_detection_demo():
        print("⚠️ Detection demo failed, but you can still try the interfaces!")

    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Try the GUI: python interface.py")
    print("2. Train on real data: python train_meowtrix.py real_dir fake_dir model.joblib")
    print("3. Use CLI: python interface.py detect image.jpg --model model.joblib")

if __name__ == "__main__":
    main()

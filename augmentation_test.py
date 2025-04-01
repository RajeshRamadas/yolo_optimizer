"""
Test data augmentation on sample images and visualize the results.

Usage:
    python test_augmentation.py --config nas_config_augmented.yaml --data path/to/data.yaml --image path/to/sample/image.jpg --output augmented_samples
"""

import os
import cv2
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from data_augmentation import (
    get_augmentation_params,
    create_augmentation_config,
    configure_additional_augmentations
)
from config_utils import load_yaml_config


def visualize_augmented_batch(model, sample_image, output_dir, num_samples=9):
    """Visualize augmented batch from the model's dataloader"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a dataloader and extract one batch
    dataloader = model.trainer.train_loader
    
    # Create a figure to display augmented images
    plt.figure(figsize=(15, 15))
    
    # Display original image
    plt.subplot(num_samples + 1, 1, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(sample_image), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Create a plot with augmented images
    aug_fig = plt.figure(figsize=(20, 20))
    
    # Process one batch from the dataloader and display images
    for i, batch in enumerate(dataloader):
        if i == 0:
            # Get images from batch (different for different YOLO versions)
            if isinstance(batch, (tuple, list)) and len(batch) > 0:
                # Newer YOLO versions return (img, targets, path, ...)
                images = batch[0]
            elif hasattr(batch, 'get'):
                # Dictionary format
                images = batch.get('img', batch.get('images', None))
            else:
                # Direct format
                images = batch
            
            # Display images
            n = min(num_samples, len(images))
            for j in range(n):
                # Convert tensor to numpy array for display
                img = images[j].permute(1, 2, 0).cpu().numpy()
                
                # Normalize for display
                img = (img - img.min()) / (img.max() - img.min())
                
                plt.subplot(3, 3, j + 1)
                plt.imshow(img)
                plt.title(f'Augmented Sample {j+1}')
                plt.axis('off')
            
            # Save the figure
            aug_fig.savefig(os.path.join(output_dir, 'augmented_samples.png'))
            plt.close(aug_fig)
            break


def apply_individual_augmentations(image_path, config, output_dir):
    """Apply individual augmentations to an image and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to RGB for display
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Save original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original.png'))
    plt.close()
    
    # Get augmentation parameters
    aug_params = get_augmentation_params(config)
    if not aug_params:
        print("No augmentation parameters found in config")
        return
    
    # Try to set up albumentations
    try:
        import albumentations as A
        
        # Create individual augmentations for demonstration
        augmentations = [
            ('Horizontal Flip', A.HorizontalFlip(p=1.0)),
            ('Vertical Flip', A.VerticalFlip(p=1.0)),
            ('Random Rotate', A.Rotate(limit=20, p=1.0)),
            ('Random Brightness Contrast', A.RandomBrightnessContrast(p=1.0)),
            ('Motion Blur', A.MotionBlur(p=1.0)),
            ('ISO Noise', A.ISONoise(p=1.0)),
            ('CLAHE', A.CLAHE(p=1.0)),
            ('Random Shadow', A.RandomShadow(p=1.0)),
            ('Cutout', A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, p=1.0))
        ]
        
        # Apply each augmentation and save
        for name, aug in augmentations:
            transform = A.Compose([aug])
            augmented = transform(image=original_image_rgb)['image']
            
            plt.figure(figsize=(10, 10))
            plt.imshow(augmented)
            plt.title(name)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}.png'))
            plt.close()
        
        # Create a mosaic-like composite of 4 augmentations
        composite_img = np.zeros((original_image_rgb.shape[0] * 2, original_image_rgb.shape[1] * 2, 3), dtype=np.uint8)
        
        # Apply 4 different augmentations
        aug_transforms = [
            A.Compose([A.HorizontalFlip(p=1.0)]),
            A.Compose([A.RandomBrightnessContrast(p=1.0)]),
            A.Compose([A.MotionBlur(p=1.0)]),
            A.Compose([A.RandomShadow(p=1.0)])
        ]
        
        # Fill the composite image
        aug_images = [transform(image=original_image_rgb)['image'] for transform in aug_transforms]
        
        composite_img[0:original_image_rgb.shape[0], 0:original_image_rgb.shape[1]] = aug_images[0]
        composite_img[0:original_image_rgb.shape[0], original_image_rgb.shape[1]:] = aug_images[1]
        composite_img[original_image_rgb.shape[0]:, 0:original_image_rgb.shape[1]] = aug_images[2]
        composite_img[original_image_rgb.shape[0]:, original_image_rgb.shape[1]:] = aug_images[3]
        
        # Save the composite
        plt.figure(figsize=(15, 15))
        plt.imshow(composite_img)
        plt.title('Mosaic-like Composite')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'composite.png'))
        plt.close()
        
    except ImportError:
        print("Albumentations not installed. Cannot visualize individual augmentations.")
    except Exception as e:
        print(f"Error in applying augmentations: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test YOLO data augmentation')
    parser.add_argument('--config', type=str, default='nas_config_augmented.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--image', type=str, required=True, help='Path to sample image')
    parser.add_argument('--output', type=str, default='augmented_samples', help='Output directory')
    parser.add_argument('--model', type=str, default='yolov8n.yaml', help='Model to use')
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Testing augmentations with config: {args.config}")
    print(f"Using image: {args.image}")
    
    # Apply individual augmentations
    individual_output = os.path.join(args.output, 'individual')
    apply_individual_augmentations(args.image, config, individual_output)
    
    # Initialize model with augmentation from config
    try:
        # Initialize model
        model = YOLO(args.model)
        
        # Set up training args with augmentation
        aug_params = get_augmentation_params(config)
        
        if aug_params:
            # Create temporary augmentation config
            aug_dir = os.path.join(args.output, 'config')
            os.makedirs(aug_dir, exist_ok=True)
            create_augmentation_config(aug_params, aug_dir)
            
            # Start training with augmentation to get a dataloader
            print("Starting a training run to get augmented samples...")
            model.train(
                data=args.data,
                epochs=1,  # Just one epoch
                imgsz=640,
                batch=4,
                augment=True,
                verbose=False,
                resume=False,
                seed=42
            )
            
            # Visualize batch with augmentations
            batch_output = os.path.join(args.output, 'batch')
            visualize_augmented_batch(model, args.image, batch_output)
            
            print(f"Augmentation test complete. Results saved to {args.output}")
        else:
            print("No augmentation parameters found in config")
    
    except Exception as e:
        print(f"Error during augmentation test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

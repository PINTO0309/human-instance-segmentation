"""Validate teacher model inference with actual images."""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.human_edge_detection.advanced.unet_decoder_distillation import DistillationUNetWrapper
import os
import glob

def validate_teacher_model():
    """Validate teacher model inference on sample images."""
    
    print("="*60)
    print("Teacher Model Validation")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model wrapper
    print("Loading teacher and student models...")
    model_wrapper = DistillationUNetWrapper(
        student_encoder="timm-efficientnet-b0",
        teacher_checkpoint_path="ext_extractor/2020-09-23a.pth",
        freeze_teacher=True
    ).to(device)
    
    # Set to eval mode
    model_wrapper.eval()
    
    # Load sample images
    image_paths = glob.glob("data/images/val2017/*.jpg")[:5]  # First 5 validation images
    
    # Define transform with proper ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create figure for visualization
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4*len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_paths):
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Load and preprocess image
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            student_output, teacher_output = model_wrapper(img_tensor)
        
        # Convert to probabilities
        student_prob = torch.sigmoid(student_output).squeeze().cpu().numpy()
        teacher_prob = torch.sigmoid(teacher_output).squeeze().cpu().numpy()
        
        print(f"  Teacher output range: [{teacher_output.min():.3f}, {teacher_output.max():.3f}]")
        print(f"  Teacher prob range: [{teacher_prob.min():.3f}, {teacher_prob.max():.3f}]")
        print(f"  Student output range: [{student_output.min():.3f}, {student_output.max():.3f}]")
        print(f"  Student prob range: [{student_prob.min():.3f}, {student_prob.max():.3f}]")
        
        # Visualize
        # Original image
        img_show = np.array(img_pil.resize((640, 640)))
        axes[idx, 0].imshow(img_show)
        axes[idx, 0].set_title('Input Image')
        axes[idx, 0].axis('off')
        
        # Teacher prediction
        axes[idx, 1].imshow(teacher_prob, cmap='gray', vmin=0, vmax=1)
        axes[idx, 1].set_title(f'Teacher (B3)\nMean: {teacher_prob.mean():.3f}')
        axes[idx, 1].axis('off')
        
        # Student prediction
        axes[idx, 2].imshow(student_prob, cmap='gray', vmin=0, vmax=1)
        axes[idx, 2].set_title(f'Student (B0)\nMean: {student_prob.mean():.3f}')
        axes[idx, 2].axis('off')
        
        # Difference
        diff = np.abs(teacher_prob - student_prob)
        axes[idx, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
        axes[idx, 3].set_title(f'Difference\nMean: {diff.mean():.3f}')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('teacher_validation.png', dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: teacher_validation.png")
    
    # Test with different normalization scenarios
    print("\n" + "="*60)
    print("Testing normalization impact:")
    print("="*60)
    
    # Create test tensor (white image)
    test_img = torch.ones(1, 3, 640, 640).to(device)
    
    # Test 1: No normalization (raw [0, 1] range)
    with torch.no_grad():
        _, teacher_no_norm = model_wrapper.teacher(test_img), model_wrapper.teacher(test_img)
        teacher_prob_no_norm = torch.sigmoid(teacher_no_norm)
    print(f"\n1. Without normalization (input [0, 1]):")
    print(f"   Output range: [{teacher_no_norm.min():.3f}, {teacher_no_norm.max():.3f}]")
    print(f"   Prob range: [{teacher_prob_no_norm.min():.3f}, {teacher_prob_no_norm.max():.3f}]")
    
    # Test 2: With ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    test_img_norm = (test_img - mean) / std
    
    with torch.no_grad():
        _, teacher_norm = model_wrapper.teacher(test_img_norm), model_wrapper.teacher(test_img_norm)
        teacher_prob_norm = torch.sigmoid(teacher_norm)
    print(f"\n2. With ImageNet normalization:")
    print(f"   Output range: [{teacher_norm.min():.3f}, {teacher_norm.max():.3f}]")
    print(f"   Prob range: [{teacher_prob_norm.min():.3f}, {teacher_prob_norm.max():.3f}]")
    
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)


if __name__ == "__main__":
    validate_teacher_model()
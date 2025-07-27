"""Investigate the pretrained model's expected input format and behavior."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import the pretrained model classes
from src.human_edge_detection.advanced.hierarchical_segmentation_unet import (
    PreTrainedPeopleSegmentationUNet
)

def check_pretrained_weights():
    """Check what's in the pretrained weights file."""
    weights_path = "ext_extractor/2020-09-23a.pth"
    
    print("=== Checking Pretrained Weights ===")
    try:
        # Load the checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check for model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
                
            print(f"\nNumber of parameters: {len(state_dict)}")
            
            # Show first few parameter names and shapes
            print("\nFirst 10 parameters:")
            for i, (name, param) in enumerate(state_dict.items()):
                if i >= 10:
                    break
                print(f"  {name}: {param.shape}")
                
            # Check for normalization parameters
            print("\nNormalization-related parameters:")
            for name, param in state_dict.items():
                if 'norm' in name.lower() or 'bn' in name.lower() or 'mean' in name.lower() or 'std' in name.lower():
                    print(f"  {name}: {param.shape}")
                    
        else:
            print(f"Checkpoint is not a dict, it's: {type(checkpoint)}")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def test_different_preprocessing():
    """Test the pretrained model with different preprocessing methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = PreTrainedPeopleSegmentationUNet(
        in_channels=3,
        classes=1,
        pretrained_weights_path="ext_extractor/2020-09-23a.pth"
    ).to(device)
    model.eval()
    
    # Create a test image with a clear person-like shape
    img_size = 512
    test_img = np.ones((img_size, img_size, 3), dtype=np.float32) * 0.8
    
    # Draw a simple person shape
    # Head
    center_x, center_y = img_size // 2, img_size // 3
    head_radius = 40
    y, x = np.ogrid[:img_size, :img_size]
    head_mask = (x - center_x)**2 + (y - center_y)**2 <= head_radius**2
    test_img[head_mask] = [0.2, 0.3, 0.4]
    
    # Body
    body_top = center_y + head_radius
    body_width = 80
    body_height = 150
    body_left = center_x - body_width // 2
    body_right = center_x + body_width // 2
    body_bottom = body_top + body_height
    test_img[body_top:body_bottom, body_left:body_right] = [0.3, 0.4, 0.5]
    
    # Test different preprocessing methods
    preprocessing_methods = {
        'no_norm': lambda x: x,  # Raw values [0, 1]
        'imagenet_norm': lambda x: (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]),
        'simple_norm': lambda x: (x - 0.5) / 0.5,  # [-1, 1]
        'zero_mean': lambda x: x - x.mean(),
        'unit_std': lambda x: (x - x.mean()) / (x.std() + 1e-6),
    }
    
    results = {}
    
    print("\n=== Testing Different Preprocessing Methods ===")
    for name, preprocess_fn in preprocessing_methods.items():
        # Preprocess
        img_preprocessed = preprocess_fn(test_img.copy())
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_preprocessed).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Get model output
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
        
        results[name] = {
            'output_range': [output.min().item(), output.max().item()],
            'output_mean': output.mean().item(),
            'prob_mean': prob.mean().item(),
            'prob_max': prob.max().item()
        }
        
        print(f"\n{name}:")
        print(f"  Input range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Output mean: {output.mean():.3f}")
        print(f"  Prob mean: {prob.mean():.3f}, max: {prob.max():.3f}")
    
    # Find best preprocessing
    best_method = max(results.items(), key=lambda x: x[1]['prob_max'])[0]
    print(f"\nBest preprocessing method: {best_method}")
    
    # Visualize results for best method
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original image
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title('Test Image')
    axes[0, 0].axis('off')
    
    # Show results for top 3 methods
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['prob_max'], reverse=True)[:3]
    
    for i, (name, result) in enumerate(sorted_methods):
        img_preprocessed = preprocessing_methods[name](test_img.copy())
        img_tensor = torch.from_numpy(img_preprocessed).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
        
        # Show output
        ax = axes[(i+1)//3, (i+1)%3]
        im = ax.imshow(prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{name}\nmax prob: {prob.max():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png')
    print("\nVisualization saved as 'preprocessing_comparison.png'")

def check_model_architecture():
    """Check the expected architecture of the pretrained model."""
    print("\n=== Checking Model Architecture ===")
    
    # Try to load model with minimal wrapper
    import torch.nn as nn
    
    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Minimal UNet structure that might match
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x
    
    # Load weights and check structure
    weights = torch.load("ext_extractor/2020-09-23a.pth", map_location='cpu')
    
    # Check if it's a state dict
    if isinstance(weights, dict) and 'state_dict' not in weights and 'model' not in weights:
        # Direct state dict
        print("Weights appear to be a direct state dict")
        print(f"Total parameters: {len(weights)}")
        
        # Group parameters by prefix
        prefixes = {}
        for name in weights.keys():
            prefix = name.split('.')[0]
            if prefix not in prefixes:
                prefixes[prefix] = 0
            prefixes[prefix] += 1
            
        print("\nParameter groups:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} parameters")

def main():
    print("=== Investigating Pretrained Model ===\n")
    
    # 1. Check weights file
    check_pretrained_weights()
    
    # 2. Test different preprocessing
    print("\n" + "="*50 + "\n")
    test_different_preprocessing()
    
    # 3. Check architecture
    print("\n" + "="*50 + "\n")
    check_model_architecture()

if __name__ == "__main__":
    main()
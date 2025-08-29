#!/usr/bin/env python3
"""Analyze how temperature decay affects KL Loss during training."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def compute_kl_divergence(student_logit, teacher_logit, temperature):
    """Compute KL divergence for binary segmentation with temperature scaling.
    
    This mimics the actual loss computation in UNetDistillationLoss.
    """
    # Apply temperature scaling and sigmoid
    student_soft = torch.sigmoid(student_logit / temperature)
    teacher_soft = torch.sigmoid(teacher_logit / temperature)
    
    # Clamp for numerical stability
    eps = 1e-5
    student_soft = torch.clamp(student_soft, eps, 1.0 - eps)
    teacher_soft = torch.clamp(teacher_soft, eps, 1.0 - eps)
    
    # KL divergence: KL(p||q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
    term1 = teacher_soft * (torch.log(teacher_soft + eps) - torch.log(student_soft + eps))
    term2 = (1 - teacher_soft) * (torch.log(1 - teacher_soft + eps) - torch.log(1 - student_soft + eps))
    
    kl_loss = (term1 + term2).mean()
    return kl_loss.item()

def analyze_temperature_effect():
    """Analyze the effect of temperature on KL Loss."""
    
    # Create sample logits representing different scenarios
    scenarios = [
        ("Similar predictions", 2.0, 2.5),  # Teacher and student agree
        ("Moderate difference", 2.0, 0.5),  # Some disagreement
        ("Large difference", 3.0, -1.0),    # Strong disagreement
        ("Opposite predictions", 2.0, -2.0), # Complete disagreement
    ]
    
    temperatures = np.linspace(1.0, 10.0, 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Effect of Temperature on KL Divergence Loss', fontsize=14, fontweight='bold')
    
    for idx, (scenario_name, teacher_logit, student_logit) in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        
        # Convert to tensors
        teacher_tensor = torch.tensor(teacher_logit, dtype=torch.float32)
        student_tensor = torch.tensor(student_logit, dtype=torch.float32)
        
        # Compute KL losses for different temperatures
        kl_losses = []
        teacher_probs = []
        student_probs = []
        
        for temp in temperatures:
            kl_loss = compute_kl_divergence(student_tensor, teacher_tensor, temp)
            kl_losses.append(kl_loss)
            
            # Also track probabilities for visualization
            teacher_prob = torch.sigmoid(teacher_tensor / temp).item()
            student_prob = torch.sigmoid(student_tensor / temp).item()
            teacher_probs.append(teacher_prob)
            student_probs.append(student_prob)
        
        # Plot KL loss
        ax.plot(temperatures, kl_losses, 'b-', linewidth=2, label='KL Loss')
        ax.set_xlabel('Temperature', fontsize=10)
        ax.set_ylabel('KL Divergence', fontsize=10)
        ax.set_title(f'{scenario_name}\n(Teacher: {teacher_logit:.1f}, Student: {student_logit:.1f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add secondary y-axis for probabilities
        ax2 = ax.twinx()
        ax2.plot(temperatures, teacher_probs, 'g--', alpha=0.5, label='Teacher prob')
        ax2.plot(temperatures, student_probs, 'r--', alpha=0.5, label='Student prob')
        ax2.set_ylabel('Probability', fontsize=10, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(loc='upper right', framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig('temperature_kl_effect.png', dpi=150, bbox_inches='tight')
    print("Saved analysis to temperature_kl_effect.png")
    
    # Print summary
    print("\n" + "="*60)
    print("TEMPERATURE DECAY EFFECTS ON KL LOSS:")
    print("="*60)
    print("\n1. HIGH TEMPERATURE (Early Training, T=10):")
    print("   - Softens probability distributions (closer to 0.5)")
    print("   - KL loss is SMALLER for all prediction differences")
    print("   - More forgiving of student mistakes")
    print("   - Allows gradual learning from teacher")
    
    print("\n2. LOW TEMPERATURE (Late Training, T=1):")
    print("   - Sharpens probability distributions (closer to 0 or 1)")
    print("   - KL loss is LARGER for prediction differences")
    print("   - More strict about matching teacher exactly")
    print("   - Forces precise alignment with teacher")
    
    print("\n3. TRAINING DYNAMICS:")
    print("   - Early epochs (T=10): Student learns general patterns")
    print("   - Middle epochs (T=5): Balance between soft and hard targets")
    print("   - Late epochs (T=1): Fine-tuning to match teacher precisely")
    
    print("\n4. PRACTICAL IMPLICATIONS:")
    print("   - Temperature decay prevents early overfitting")
    print("   - Gradual transition from soft to hard knowledge transfer")
    print("   - Better convergence and stability")
    
    # Create temporal evolution plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Training Evolution with Temperature Scheduling', fontsize=14, fontweight='bold')
    
    epochs = np.arange(0, 100)
    # Linear temperature decay from 10 to 1
    temps_linear = 10 - (epochs / 100) * 9
    
    # Different prediction error scenarios over time
    # Assume student gradually improves
    teacher_logit = 2.0
    
    for idx, (title, initial_diff) in enumerate([
        ("Fast Learner", 3.0),
        ("Average Learner", 2.0),
        ("Slow Learner", 1.0)
    ]):
        ax = axes2[idx]
        
        kl_losses_over_time = []
        student_improvements = []
        
        for epoch in epochs:
            # Student improves over time
            improvement_rate = initial_diff / 100
            student_logit = -1.0 + improvement_rate * epoch
            student_improvements.append(student_logit)
            
            # Current temperature
            temp = temps_linear[epoch]
            
            # Compute KL loss
            kl_loss = compute_kl_divergence(
                torch.tensor(student_logit),
                torch.tensor(teacher_logit),
                temp
            )
            kl_losses_over_time.append(kl_loss)
        
        # Plot
        ax.plot(epochs, kl_losses_over_time, 'b-', linewidth=2, label='KL Loss')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('KL Loss', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add temperature on secondary axis
        ax2 = ax.twinx()
        ax2.plot(epochs, temps_linear, 'r--', alpha=0.5, label='Temperature')
        ax2.set_ylabel('Temperature', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('temperature_training_evolution.png', dpi=150, bbox_inches='tight')
    print("\nSaved training evolution to temperature_training_evolution.png")

if __name__ == "__main__":
    analyze_temperature_effect()
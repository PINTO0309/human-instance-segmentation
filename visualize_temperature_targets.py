#!/usr/bin/env python3
"""Visualize the relationship between Temperature, Soft Targets, and Hard Targets."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def sigmoid_with_temperature(logit, temperature):
    """Apply sigmoid with temperature scaling."""
    return torch.sigmoid(torch.tensor(logit) / temperature).item()

def create_temperature_visualization():
    """Create comprehensive visualization of temperature effects."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Define a sample logit value (teacher's output before sigmoid)
    logit = 2.0  # Positive logit → teacher predicts "foreground"
    
    # Different temperature values
    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # ========== 1. Temperature Effect on Single Prediction ==========
    ax1 = plt.subplot(2, 3, 1)
    
    probs = [sigmoid_with_temperature(logit, t) for t in temperatures]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(temperatures)))
    
    bars = ax1.bar(range(len(temperatures)), probs, color=colors)
    ax1.set_xticks(range(len(temperatures)))
    ax1.set_xticklabels([f'T={t}' for t in temperatures])
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title(f'Sigmoid Output for Logit={logit}\nat Different Temperatures', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    # ========== 2. Soft vs Hard Targets Comparison ==========
    ax2 = plt.subplot(2, 3, 2)
    
    # Define multiple logits for a mini-batch
    logits = np.array([-3, -1, 0, 1, 3])
    x_pos = np.arange(len(logits))
    
    # Calculate probabilities at different temperatures
    hard_probs = [sigmoid_with_temperature(l, 1.0) for l in logits]  # T=1 (hard)
    soft_probs = [sigmoid_with_temperature(l, 10.0) for l in logits]  # T=10 (soft)
    
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, hard_probs, width, label='Hard Target (T=1)', color='red', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, soft_probs, width, label='Soft Target (T=10)', color='blue', alpha=0.7)
    
    ax2.set_xlabel('Logit Value', fontsize=11)
    ax2.set_ylabel('Probability', fontsize=11)
    ax2.set_title('Hard vs Soft Targets', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(logits)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    # ========== 3. Temperature Continuum ==========
    ax3 = plt.subplot(2, 3, 3)
    
    temps_continuous = np.linspace(0.5, 20, 100)
    logit_values = [-2, -1, 0, 1, 2]
    
    for logit_val in logit_values:
        probs_continuous = [sigmoid_with_temperature(logit_val, t) for t in temps_continuous]
        ax3.plot(temps_continuous, probs_continuous, linewidth=2, label=f'Logit={logit_val}')
    
    ax3.set_xlabel('Temperature', fontsize=11)
    ax3.set_ylabel('Probability', fontsize=11)
    ax3.set_title('Probability vs Temperature\nfor Different Logits', fontsize=12, fontweight='bold')
    ax3.legend(loc='right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    
    # ========== 4. Information Content ==========
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate entropy for different temperatures
    temps_entropy = np.linspace(0.5, 20, 50)
    entropies = []
    
    for temp in temps_entropy:
        # Calculate probabilities for a set of diverse logits
        test_logits = np.linspace(-3, 3, 7)
        probs = [sigmoid_with_temperature(l, temp) for l in test_logits]
        
        # Calculate average entropy
        entropy = 0
        for p in probs:
            if p > 0 and p < 1:
                entropy -= p * np.log(p) + (1-p) * np.log(1-p)
        entropies.append(entropy / len(probs))
    
    ax4.plot(temps_entropy, entropies, linewidth=3, color='purple')
    ax4.fill_between(temps_entropy, entropies, alpha=0.3, color='purple')
    ax4.set_xlabel('Temperature', fontsize=11)
    ax4.set_ylabel('Average Entropy', fontsize=11)
    ax4.set_title('Information Content\n(Higher = More Uncertain)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add annotations
    ax4.annotate('Hard Targets\n(Confident)', xy=(1, entropies[2]), xytext=(3, entropies[2]-0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    ax4.annotate('Soft Targets\n(Uncertain)', xy=(15, entropies[-10]), xytext=(12, entropies[-10]+0.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue', fontweight='bold')
    
    # ========== 5. Gradient Magnitude ==========
    ax5 = plt.subplot(2, 3, 5)
    
    # Simulate gradient magnitude for different temperatures
    temps_grad = np.linspace(0.5, 20, 50)
    gradient_mags = []
    
    for temp in temps_grad:
        # Gradient of sigmoid is p(1-p)/T
        # Higher temperature → smaller gradients
        test_logit = 1.0
        p = sigmoid_with_temperature(test_logit, temp)
        grad_mag = p * (1 - p) / temp
        gradient_mags.append(grad_mag)
    
    ax5.plot(temps_grad, gradient_mags, linewidth=3, color='green')
    ax5.fill_between(temps_grad, gradient_mags, alpha=0.3, color='green')
    ax5.set_xlabel('Temperature', fontsize=11)
    ax5.set_ylabel('Gradient Magnitude', fontsize=11)
    ax5.set_title('Learning Signal Strength\n(Gradient Flow)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. Practical Example ==========
    ax6 = plt.subplot(2, 3, 6)
    
    # Show a practical segmentation example
    epochs = np.arange(0, 100)
    
    # Temperature schedule (linear decay from 10 to 1)
    temp_schedule = 10 - (epochs / 100) * 9
    
    # Simulated accuracy improvement
    accuracy_soft = 50 + 45 * (1 - np.exp(-epochs/30))  # Smooth improvement
    accuracy_hard = 50 + 35 * (1 - np.exp(-epochs/30)) + np.random.normal(0, 3, len(epochs))  # More volatile
    
    ax6.plot(epochs, accuracy_soft, 'b-', linewidth=2, label='With Temp Scheduling', alpha=0.8)
    ax6.plot(epochs, accuracy_hard, 'r--', linewidth=2, label='Without Temp (T=1)', alpha=0.8)
    
    ax6_twin = ax6.twinx()
    ax6_twin.plot(epochs, temp_schedule, 'g:', linewidth=2, label='Temperature', alpha=0.6)
    ax6_twin.set_ylabel('Temperature', color='green', fontsize=11)
    ax6_twin.tick_params(axis='y', labelcolor='green')
    
    ax6.set_xlabel('Epoch', fontsize=11)
    ax6.set_ylabel('Accuracy (%)', fontsize=11)
    ax6.set_title('Training with Temperature Scheduling', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.suptitle('Temperature, Soft Targets, and Hard Targets Explained', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('temperature_targets_explained.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to temperature_targets_explained.png")

def create_conceptual_explanation():
    """Create a conceptual diagram explaining the relationship."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Conceptual Understanding: Temperature in Knowledge Distillation', 
                 fontsize=14, fontweight='bold')
    
    # ========== Panel 1: Hard Target ==========
    ax1 = axes[0]
    ax1.set_title('Hard Target (T=1)\n"Binary Decision"', fontsize=12, fontweight='bold')
    
    # Create binary-like distribution
    categories = ['Background', 'Foreground']
    hard_values = [0.05, 0.95]  # Very confident
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax1.bar(categories, hard_values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Probability', fontsize=11)
    
    # Add annotations
    ax1.text(0.5, 0.5, 'Clear Cut Decision\n• High Confidence\n• Less Information\n• Can be Overconfident', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for bar, val in zip(bars, hard_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ========== Panel 2: Soft Target ==========
    ax2 = axes[1]
    ax2.set_title('Soft Target (T=10)\n"Uncertain/Nuanced"', fontsize=12, fontweight='bold')
    
    # Create softer distribution
    soft_values = [0.35, 0.65]  # Less confident
    
    bars = ax2.bar(categories, soft_values, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Probability', fontsize=11)
    
    # Add annotations
    ax2.text(0.5, 0.5, 'Nuanced Decision\n• Lower Confidence\n• More Information\n• Better for Learning', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    for bar, val in zip(bars, soft_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # ========== Panel 3: Temperature Effect ==========
    ax3 = axes[2]
    ax3.set_title('Temperature Control', fontsize=12, fontweight='bold')
    
    # Show transformation
    temps = [1, 2, 5, 10, 20]
    probs = []
    logit = 2.0  # Fixed logit
    
    for t in temps:
        p = sigmoid_with_temperature(logit, t)
        probs.append(p)
    
    # Create color gradient
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(temps)))
    
    bars = ax3.bar(range(len(temps)), probs, color=colors, edgecolor='black', linewidth=1)
    ax3.set_xticks(range(len(temps)))
    ax3.set_xticklabels([f'T={t}' for t in temps])
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Foreground Probability', fontsize=11)
    ax3.set_xlabel('Temperature Setting', fontsize=11)
    
    # Add arrow showing progression
    ax3.annotate('', xy=(4.2, 0.55), xytext=(-0.2, 0.88),
                arrowprops=dict(arrowstyle='->', lw=3, color='purple', alpha=0.6))
    ax3.text(2, 0.9, 'Hard→Soft', ha='center', fontsize=11, 
             color='purple', fontweight='bold', rotation=-8)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('temperature_concept_explained.png', dpi=150, bbox_inches='tight')
    print("Saved conceptual diagram to temperature_concept_explained.png")

def print_explanation():
    """Print a clear textual explanation."""
    
    print("\n" + "="*70)
    print("TEMPERATURE, SOFT TARGETS, AND HARD TARGETS EXPLAINED")
    print("="*70)
    
    print("\n📊 BASIC CONCEPT:")
    print("-"*40)
    print("Temperature (T) controls how 'confident' or 'uncertain' predictions are:")
    print("• Logit → Sigmoid(Logit/T) → Probability")
    print("• Higher T → More uncertain (soft)")
    print("• Lower T → More confident (hard)")
    
    print("\n🎯 HARD TARGETS (T=1 or low):")
    print("-"*40)
    print("• Clear binary decisions (0.99 vs 0.01)")
    print("• High confidence")
    print("• Less nuanced information")
    print("• Example: 'This pixel is DEFINITELY foreground!'")
    print("• Risk: Overconfidence, harder to learn from")
    
    print("\n☁️ SOFT TARGETS (T=10 or high):")
    print("-"*40)
    print("• Uncertain predictions (0.65 vs 0.35)")
    print("• Lower confidence")
    print("• Rich, nuanced information")
    print("• Example: 'This pixel is PROBABLY foreground...'")
    print("• Benefit: Easier to learn from, more informative")
    
    print("\n🔄 TEMPERATURE SCHEDULING IN TRAINING:")
    print("-"*40)
    print("Early Training (T=10):")
    print("  → Soft targets")
    print("  → Student learns general patterns")
    print("  → Forgiving of mistakes")
    print("")
    print("Mid Training (T=5):")
    print("  → Balanced targets")
    print("  → Refining understanding")
    print("")
    print("Late Training (T=1):")
    print("  → Hard targets")
    print("  → Fine-tuning details")
    print("  → Precise matching")
    
    print("\n💡 ANALOGY:")
    print("-"*40)
    print("Think of it like teaching:")
    print("• Soft Targets = Gentle teacher saying 'Maybe try this way...'")
    print("• Hard Targets = Strict teacher saying 'This is RIGHT, that is WRONG!'")
    print("• Temperature Scheduling = Starting gentle, becoming stricter over time")
    
    print("\n🚀 WHY IT WORKS:")
    print("-"*40)
    print("1. Soft targets contain more information about relative probabilities")
    print("2. Gradients flow better with soft targets (avoid saturation)")
    print("3. Progressive hardening prevents early overfitting")
    print("4. Natural curriculum: easy → hard")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Generating visualizations...")
    create_temperature_visualization()
    create_conceptual_explanation()
    print_explanation()
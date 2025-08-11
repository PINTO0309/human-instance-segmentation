#!/usr/bin/env python3
"""Clarify the effect of temperature on distillation vs ground truth loss."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def analyze_loss_components():
    """Analyze how temperature affects different loss components."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temperature効果の正しい理解：教師モデル vs 元画像の影響', 
                 fontsize=14, fontweight='bold')
    
    # ========== 1. KL Loss vs Temperature ==========
    ax1 = axes[0, 0]
    
    # Simulate KL loss for different prediction differences
    temps = np.linspace(1, 10, 50)
    
    # Different scenarios of student-teacher disagreement
    scenarios = [
        ("Small diff", 2.0, 1.8),   # Teacher and student almost agree
        ("Medium diff", 2.0, 1.0),  # Some disagreement
        ("Large diff", 2.0, 0.0),   # Large disagreement
    ]
    
    for name, teacher_logit, student_logit in scenarios:
        kl_losses = []
        for temp in temps:
            # Calculate KL divergence
            student_soft = torch.sigmoid(torch.tensor(student_logit) / temp)
            teacher_soft = torch.sigmoid(torch.tensor(teacher_logit) / temp)
            
            eps = 1e-7
            student_soft = torch.clamp(student_soft, eps, 1-eps)
            teacher_soft = torch.clamp(teacher_soft, eps, 1-eps)
            
            kl = teacher_soft * torch.log(teacher_soft/student_soft) + \
                 (1-teacher_soft) * torch.log((1-teacher_soft)/(1-student_soft))
            kl_losses.append(kl.item())
        
        ax1.plot(temps, kl_losses, linewidth=2, label=name)
    
    ax1.set_xlabel('Temperature', fontsize=11)
    ax1.set_ylabel('KL Loss (教師モデルとの差)', fontsize=11)
    ax1.set_title('Temperature↓ → KL Loss↑\n(教師モデルの影響が強くなる)', 
                  fontsize=12, fontweight='bold', color='red')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Show decreasing temperature
    
    # ========== 2. Loss Balance ==========
    ax2 = axes[0, 1]
    
    # In typical distillation: Loss = α*KL_loss + (1-α)*GT_loss
    # But KL_loss magnitude changes with temperature!
    
    epochs = np.arange(0, 100)
    temps_schedule = 10 - (epochs/100) * 9  # Linear decay from 10 to 1
    
    # Simulate relative loss magnitudes
    kl_weight = 0.7  # alpha
    gt_weight = 0.3  # 1-alpha
    
    # KL loss increases as temperature decreases
    kl_loss_magnitude = 0.5 / temps_schedule  # Inversely proportional
    gt_loss_magnitude = np.ones_like(epochs) * 0.3  # Constant
    
    # Total weighted contributions
    kl_contribution = kl_weight * kl_loss_magnitude
    gt_contribution = gt_weight * gt_loss_magnitude
    
    ax2.plot(epochs, kl_contribution, 'b-', linewidth=2, label='教師モデルの影響 (KL×α)')
    ax2.plot(epochs, gt_contribution, 'g-', linewidth=2, label='元画像の影響 (GT×(1-α))')
    ax2.fill_between(epochs, 0, kl_contribution, alpha=0.3, color='blue')
    ax2.fill_between(epochs, 0, gt_contribution, alpha=0.3, color='green')
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss への寄与度', fontsize=11)
    ax2.set_title('学習が進むと教師モデルの影響が強くなる\n(Temperature↓ → KL Loss↑)', 
                  fontsize=12, fontweight='bold', color='red')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add temperature on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, temps_schedule, 'r:', linewidth=1, alpha=0.5)
    ax2_twin.set_ylabel('Temperature', color='red', fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # ========== 3. Gradient Flow ==========
    ax3 = axes[1, 0]
    
    temps = np.linspace(1, 10, 50)
    
    # Gradient magnitude for KL loss
    gradient_from_teacher = []
    gradient_from_gt = []
    
    for temp in temps:
        # KL gradient increases as temperature decreases
        # This is because differences become more pronounced
        kl_grad = 1.0 / temp  # Simplified: actual gradient ∝ 1/T
        gradient_from_teacher.append(kl_grad)
        
        # GT gradient stays relatively constant
        gt_grad = 0.5
        gradient_from_gt.append(gt_grad)
    
    ax3.plot(temps, gradient_from_teacher, 'b-', linewidth=3, label='教師モデルからの勾配')
    ax3.plot(temps, gradient_from_gt, 'g-', linewidth=3, label='元画像からの勾配')
    
    ax3.set_xlabel('Temperature', fontsize=11)
    ax3.set_ylabel('勾配の強さ', fontsize=11)
    ax3.set_title('Temperature↓ → 教師モデルからの学習信号↑', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Annotate
    ax3.annotate('強い教師信号', xy=(1.5, 0.6), xytext=(3, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue', fontweight='bold')
    
    # ========== 4. Conceptual Explanation ==========
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
    ❌ 間違った理解:
    「Temperature↓ → 元画像の影響↑」
    
    ✅ 正しい理解:
    「Temperature↓ → 教師モデルの影響↑」
    
    なぜ？
    
    1. Temperature が小さくなると:
       • 確率分布がシャープになる
       • 小さな差が大きな KL Loss になる
       • 教師との一致がより厳密に要求される
    
    2. 損失関数の構成:
       Loss = α × KL_Loss + (1-α) × GT_Loss
       
       • KL_Loss: 教師モデルとの差（T↓で増大）
       • GT_Loss: 元画像との差（Tに無関係）
    
    3. 実際の効果:
       初期（T=10）: ゆるく教師を真似る
       後期（T=1）: 厳密に教師を真似る
    
    元画像（Ground Truth）の影響は
    Temperature に関係なく一定！
    """
    
    ax4.text(0.1, 0.5, explanation, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('temperature_effect_clarified.png', dpi=150, bbox_inches='tight')
    print("Saved clarification to temperature_effect_clarified.png")

def print_clarification():
    """Print clear clarification of the misconception."""
    
    print("\n" + "="*70)
    print("TEMPERATURE 効果の正しい理解")
    print("="*70)
    
    print("\n❌ よくある誤解:")
    print("-"*40)
    print("「Temperature が下がると元画像（Ground Truth）の影響が強くなる」")
    print("→ これは間違いです！")
    
    print("\n✅ 正しい理解:")
    print("-"*40)
    print("「Temperature が下がると教師モデルの影響が強くなる」")
    
    print("\n📊 理由:")
    print("-"*40)
    print("1. Temperature が小さくなると（10→1）:")
    print("   • Sigmoid の出力がシャープになる（0.5→0.9など）")
    print("   • 教師と生徒の小さな差が大きな KL Loss になる")
    print("   • 教師モデルとの正確な一致が要求される")
    
    print("\n2. 損失関数の構成:")
    print("   Total Loss = α × KL_Loss(T) + (1-α) × BCE_Loss")
    print("   ")
    print("   • KL_Loss: 教師モデルとの差")
    print("     → Temperature に依存（T↓ で影響↑）")
    print("   • BCE_Loss: 元画像との差")  
    print("     → Temperature に無関係（常に一定）")
    
    print("\n🔄 学習の進行:")
    print("-"*40)
    print("初期（T=10, Epoch 1-20）:")
    print("  • ソフトな教師信号")
    print("  • 教師をゆるく真似る")
    print("  • KL Loss は小さい")
    print("")
    print("後期（T=1, Epoch 80-100）:")
    print("  • ハードな教師信号")
    print("  • 教師を厳密に真似る")
    print("  • KL Loss は大きい")
    
    print("\n💡 重要なポイント:")
    print("-"*40)
    print("• Ground Truth（元画像）の寄与は Temperature に関係なく一定")
    print("• Temperature は教師-生徒間の KL Loss のみに影響")
    print("• Temperature↓ = 教師モデルへの追従を厳密化")
    
    print("\n📈 実用的な意味:")
    print("-"*40)
    print("Temperature Scheduling により:")
    print("1. 序盤: 大まかに教師の知識を吸収")
    print("2. 終盤: 教師の細かい判断まで正確に再現")
    print("→ 段階的により精密な知識蒸留を実現")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Analyzing temperature effect on loss components...")
    analyze_loss_components()
    print_clarification()
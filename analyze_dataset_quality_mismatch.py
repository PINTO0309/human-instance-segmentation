#!/usr/bin/env python3
"""Analyze the impact of dataset quality mismatch in knowledge distillation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_dataset_mismatch_visualization():
    """Visualize the problem of dataset quality mismatch in distillation."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dataset Quality Mismatch in Knowledge Distillation', 
                 fontsize=14, fontweight='bold')
    
    # ========== 1. Dataset Quality Comparison ==========
    ax1 = axes[0, 0]
    ax1.set_title('Dataset Quality Difference', fontsize=12, fontweight='bold')
    
    # Create quality comparison bars
    datasets = ['Teacher\n(High Quality)', 'Student\n(Real World)']
    qualities = {
        'Clean': [95, 60],
        'Noise': [5, 40],
    }
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bottom_clean = [0, 0]
    bars1 = ax1.bar(x, qualities['Clean'], width, label='Clean Data', 
                    color='lightgreen', edgecolor='black')
    bars2 = ax1.bar(x, qualities['Noise'], width, bottom=qualities['Clean'],
                    label='Noisy/Ambiguous', color='lightcoral', edgecolor='black')
    
    ax1.set_ylabel('Data Composition (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim([0, 100])
    
    # Add annotations
    ax1.text(0, 50, '95%\nClean', ha='center', va='center', fontweight='bold')
    ax1.text(1, 30, '60%\nClean', ha='center', va='center', fontweight='bold')
    ax1.text(1, 80, '40%\nNoisy', ha='center', va='center', fontweight='bold', color='darkred')
    
    # ========== 2. Problem Cases ==========
    ax2 = axes[0, 1]
    ax2.set_title('Problem Cases', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    problem_text = """
    🔴 Edge Cases Teacher Never Saw:
    • Motion blur
    • Low resolution
    • Partial occlusions
    • Extreme lighting
    • Sensor noise
    
    🟡 Domain Shift Issues:
    • Teacher: Studio conditions
    • Student: Real-world chaos
    
    🔵 Overconfidence Transfer:
    • Teacher is 99% confident (wrong!)
    • Student forced to copy this
    • Temperature ↓ makes it worse
    """
    
    ax2.text(0.1, 0.5, problem_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ========== 3. Temperature Impact ==========
    ax3 = axes[0, 2]
    ax3.set_title('Impact of Temperature Decay', fontsize=12, fontweight='bold')
    
    epochs = np.arange(0, 100)
    temps = 10 - (epochs/100) * 9
    
    # Simulate accuracy with dataset mismatch
    # Good case: matching datasets
    acc_matched = 50 + 45 * (1 - np.exp(-epochs/30))
    
    # Bad case: mismatched datasets (teacher on high quality)
    # Initially benefits from teacher, then suffers
    acc_mismatched = 50 + 35 * (1 - np.exp(-epochs/25))
    # Add degradation in late training when T is low
    late_degradation = np.where(epochs > 60, 
                                (epochs - 60) * 0.2, 
                                0)
    acc_mismatched = acc_mismatched - late_degradation
    
    ax3.plot(epochs, acc_matched, 'g-', linewidth=2, 
             label='Matched Quality', alpha=0.8)
    ax3.plot(epochs, acc_mismatched, 'r-', linewidth=2, 
             label='Teacher: High Quality\nStudent: Noisy', alpha=0.8)
    
    # Add temperature
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, temps, 'b:', linewidth=1, alpha=0.5, label='Temperature')
    ax3_twin.set_ylabel('Temperature', color='blue', fontsize=10)
    ax3_twin.tick_params(axis='y', labelcolor='blue')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.3)
    
    # Mark problem region
    ax3.axvspan(60, 100, alpha=0.2, color='red', label='Problem Zone')
    ax3.annotate('Performance\nDegradation', xy=(80, 75), xytext=(85, 85),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # ========== 4. Example: Clean vs Noisy ==========
    ax4 = axes[1, 0]
    ax4.set_title('Example: Boundary Detection', fontsize=12, fontweight='bold')
    
    # Simulate clean vs noisy boundaries
    x = np.linspace(0, 10, 100)
    
    # Teacher's clean boundary
    teacher_boundary = 5 + 0.1 * np.sin(x)
    ax4.plot(x, teacher_boundary, 'g-', linewidth=3, label='Teacher (Clean Data)')
    ax4.fill_between(x, teacher_boundary - 0.2, teacher_boundary + 0.2, 
                     alpha=0.3, color='green')
    
    # Real noisy boundary
    np.random.seed(42)
    real_boundary = 5 + 0.1 * np.sin(x) + np.random.normal(0, 0.3, len(x))
    ax4.plot(x, real_boundary, 'b--', linewidth=2, alpha=0.7, 
             label='Real Data (Noisy)')
    
    # Student trying to match teacher (problematic)
    student_boundary = teacher_boundary + np.random.normal(0, 0.05, len(x))
    ax4.plot(x, student_boundary, 'r-', linewidth=2, alpha=0.8,
             label='Student (T=1, Forced to match teacher)')
    
    ax4.set_xlabel('Position', fontsize=11)
    ax4.set_ylabel('Boundary', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========== 5. Solution Strategies ==========
    ax5 = axes[1, 1]
    ax5.set_title('Mitigation Strategies', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    solution_text = """
    ✅ SOLUTIONS:
    
    1. Adaptive Temperature:
       • Detect domain shift
       • Keep T higher when mismatch detected
    
    2. Confidence Calibration:
       • Reduce teacher confidence
       • Add uncertainty modeling
    
    3. Hybrid Loss:
       • Increase GT weight (1-α) late training
       • Reduce KL weight (α) when T is low
    
    4. Data Augmentation:
       • Add noise to teacher training
       • Or clean student data
    
    5. Early Stopping:
       • Stop before overfit to teacher
       • Monitor real-world validation
    """
    
    ax5.text(0.1, 0.5, solution_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # ========== 6. Recommendation ==========
    ax6 = axes[1, 2]
    ax6.set_title('Recommendation', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    rec_text = """
    ⚠️ WARNING SIGNS:
    • Val loss ↓ but real performance ↓
    • High confidence on ambiguous cases
    • Poor generalization to new data
    
    📊 MONITORING:
    • Track both clean & noisy val sets
    • Watch for divergence late training
    • Compare teacher vs student errors
    
    🎯 BEST PRACTICE:
    When teacher has cleaner data:
    
    1. Use higher final Temperature (3-5)
    2. Increase GT weight in late training
    3. Add noise augmentation
    4. Consider stopping Temperature 
       decay at epoch 60-70
    
    Remember: Perfect teacher ≠ Perfect student
    if domains don't match!
    """
    
    ax6.text(0.05, 0.5, rec_text, fontsize=9, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('dataset_quality_mismatch.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to dataset_quality_mismatch.png")

def print_analysis():
    """Print detailed analysis of the problem."""
    
    print("\n" + "="*70)
    print("高品質データで学習した教師モデルの潜在的問題")
    print("="*70)
    
    print("\n🎯 問題の核心:")
    print("-"*40)
    print("教師モデル: 高品質・クリーンなデータで学習")
    print("生徒モデル: 実世界のノイジーなデータで運用")
    print("→ ドメインシフトが発生！")
    
    print("\n📈 学習後半（T→1）で起こること:")
    print("-"*40)
    print("1. 教師の影響が最大化")
    print("2. 教師の「理想的すぎる」判断を強制的にコピー")
    print("3. 実データの特性（ノイズ等）を無視")
    print("4. 結果: 実環境での性能劣化")
    
    print("\n🔴 具体的な悪影響の例:")
    print("-"*40)
    print("• エッジケース: 教師が見たことない劣化パターン")
    print("  → 教師: 自信満々で間違った予測")
    print("  → 生徒: それを忠実にコピー（T=1で強制）")
    print("")
    print("• ノイズ処理: ")
    print("  → 教師: ノイズを知らない → シャープな境界")
    print("  → 実データ: ノイジーな境界")
    print("  → 生徒: 不適切な境界学習")
    
    print("\n💡 解決策:")
    print("-"*40)
    print("1. Temperature Scheduling の修正:")
    print("   • 最終 Temperature を高めに (T=3~5)")
    print("   • 早めに decay を止める")
    print("")
    print("2. Loss バランスの調整:")
    print("   • 後半で GT weight (1-α) を増やす")
    print("   • KL weight (α) を減らす")
    print("")
    print("3. データの工夫:")
    print("   • 教師の学習データにノイズ追加")
    print("   • または生徒データをクリーン化")
    print("")
    print("4. アンサンブル的アプローチ:")
    print("   • 教師の知識 + 実データから学習")
    
    print("\n⚡ 実装例:")
    print("-"*40)
    print("```python")
    print("# 後半で GT の影響を増やす")
    print("if epoch > 60:")
    print("    alpha = max(0.3, alpha - 0.01)  # KL weight を減らす")
    print("    ")
    print("# Temperature decay を早めに止める")
    print("final_temperature = 3.0  # 1 ではなく 3")
    print("if epoch > 70:")
    print("    temperature = final_temperature  # これ以上下げない")
    print("```")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Analyzing dataset quality mismatch impact...")
    create_dataset_mismatch_visualization()
    print_analysis()
#!/usr/bin/env python3
"""Analyze the case where teacher has LOW quality data and student has HIGH quality data."""

import numpy as np
import matplotlib.pyplot as plt

def create_reverse_quality_visualization():
    """Visualize when teacher has lower quality data than student."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('教師:低品質 → 生徒:高品質 データでの蒸留', 
                 fontsize=14, fontweight='bold')
    
    # ========== 1. Dataset Quality Comparison (REVERSED) ==========
    ax1 = axes[0, 0]
    ax1.set_title('Dataset Quality (逆転ケース)', fontsize=12, fontweight='bold')
    
    datasets = ['Teacher\n(Low Quality)', 'Student\n(High Quality)']
    qualities = {
        'Clean': [60, 95],  # Reversed!
        'Noise': [40, 5],
    }
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x, qualities['Clean'], width, label='Clean Data', 
                    color='lightgreen', edgecolor='black')
    bars2 = ax1.bar(x, qualities['Noise'], width, bottom=qualities['Clean'],
                    label='Noisy/Ambiguous', color='lightcoral', edgecolor='black')
    
    ax1.set_ylabel('Data Composition (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.set_ylim([0, 100])
    
    ax1.text(0, 30, '60%\nClean', ha='center', va='center', fontweight='bold')
    ax1.text(0, 80, '40%\nNoisy', ha='center', va='center', fontweight='bold', color='darkred')
    ax1.text(1, 50, '95%\nClean', ha='center', va='center', fontweight='bold', color='darkgreen')
    
    # ========== 2. What Happens ==========
    ax2 = axes[0, 1]
    ax2.set_title('この状況で起こること', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    situation_text = """
    🟢 POSITIVE EFFECTS:
    
    初期 (T=10, ソフトターゲット):
    • 教師の「大まかな」知識を吸収
    • ノイズの影響は平滑化される
    • 基本的なパターンを学習
    
    後期 (T=1, ハードターゲット):
    • 教師の影響は強いが...
    • 高品質GTデータが補正！
    • Loss = α×KL + (1-α)×BCE_GT
    
    🔵 なぜ問題にならないか:
    • GTデータが高品質 → 正しい方向へ導く
    • 教師のノイズ → GTが修正
    • 教師の不確実性 → 生徒が改善
    """
    
    ax2.text(0.1, 0.5, situation_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # ========== 3. Learning Dynamics ==========
    ax3 = axes[0, 2]
    ax3.set_title('学習ダイナミクス', fontsize=12, fontweight='bold')
    
    epochs = np.arange(0, 100)
    temps = 10 - (epochs/100) * 9
    
    # Teacher performance (limited by noisy data)
    teacher_perf = np.ones(100) * 82  # Capped at 82% due to noise
    
    # Student performance (can exceed teacher!)
    student_perf = 50 + 45 * (1 - np.exp(-epochs/25))
    # Student can surpass teacher thanks to better data
    student_perf = np.minimum(student_perf, 95)  # Cap at 95%
    
    ax3.plot(epochs, teacher_perf, 'r--', linewidth=2, 
             label='Teacher (上限82%)', alpha=0.7)
    ax3.plot(epochs, student_perf, 'g-', linewidth=3, 
             label='Student (高品質データで改善)', alpha=0.8)
    
    # Add temperature
    ax3_twin = ax3.twinx()
    ax3_twin.plot(epochs, temps, 'b:', linewidth=1, alpha=0.5)
    ax3_twin.set_ylabel('Temperature', color='blue', fontsize=10)
    ax3_twin.tick_params(axis='y', labelcolor='blue')
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.legend(loc='center left')
    ax3.grid(True, alpha=0.3)
    
    # Mark where student surpasses teacher
    surpass_epoch = np.where(student_perf > teacher_perf)[0][0]
    ax3.axvline(x=surpass_epoch, color='gold', linestyle='--', alpha=0.7)
    ax3.annotate('生徒が教師を超える!', xy=(surpass_epoch, 85), xytext=(surpass_epoch+10, 90),
                arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                fontsize=10, color='gold', fontweight='bold')
    
    # ========== 4. Loss Components ==========
    ax4 = axes[1, 0]
    ax4.set_title('Loss成分の寄与', fontsize=12, fontweight='bold')
    
    # Show how different loss components help
    epochs_loss = np.arange(0, 100)
    
    # KL Loss: 教師から基礎を学ぶ（ノイズ含む）
    kl_contribution = 0.7 * np.exp(-epochs_loss/50)  # Decreases over time
    
    # BCE Loss: 高品質GTが正しく導く
    bce_contribution = 0.3 + 0.2 * (epochs_loss/100)  # Increases importance
    
    ax4.fill_between(epochs_loss, 0, kl_contribution, alpha=0.5, color='red', 
                     label='KL (教師の知識)')
    ax4.fill_between(epochs_loss, kl_contribution, kl_contribution + bce_contribution, 
                     alpha=0.5, color='green', label='BCE (高品質GT)')
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss Contribution', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax4.annotate('GTが主導権を取る', xy=(70, 0.6), xytext=(50, 0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')
    
    # ========== 5. Boundary Example ==========
    ax5 = axes[1, 1]
    ax5.set_title('境界検出の例', fontsize=12, fontweight='bold')
    
    x = np.linspace(0, 10, 100)
    
    # True boundary (clean)
    true_boundary = 5 + 0.1 * np.sin(x)
    ax5.plot(x, true_boundary, 'g-', linewidth=3, label='真の境界 (GT)', alpha=0.9)
    
    # Teacher's noisy understanding
    np.random.seed(42)
    teacher_boundary = true_boundary + np.random.normal(0, 0.3, len(x))
    ax5.plot(x, teacher_boundary, 'r--', linewidth=2, 
             label='教師 (ノイジー)', alpha=0.6)
    
    # Student learns better!
    student_boundary = true_boundary + np.random.normal(0, 0.05, len(x))
    ax5.plot(x, student_boundary, 'b-', linewidth=2.5, 
             label='生徒 (改善された)', alpha=0.8)
    
    ax5.set_xlabel('Position', fontsize=11)
    ax5.set_ylabel('Boundary', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ========== 6. Benefits Summary ==========
    ax6 = axes[1, 2]
    ax6.set_title('この状況の利点', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    benefits_text = """
    ✨ BENEFITS:
    
    1. 知識蒸留 + データ改善:
       • 教師の基礎知識を獲得
       • 高品質データで精錬
       • 教師を超える性能達成可能
    
    2. ノイズ耐性:
       • 教師のノイズを学習初期に吸収
       • 高品質GTで徐々に修正
       • 最終的にクリーンな出力
    
    3. Temperature Scheduling が有効:
       • T=10: ノイズを平滑化して吸収
       • T=1: GTで精密に調整
    
    💡 結論:
    教師が低品質でも問題なし！
    むしろ生徒が改善できる良い状況
    
    ⚠️ 注意点:
    • α (KL weight) を小さめに
    • GTの品質を信頼
    • 教師は「ヒント」程度に
    """
    
    ax6.text(0.05, 0.5, benefits_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('teacher_low_quality_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to teacher_low_quality_analysis.png")

def print_detailed_analysis():
    """Print detailed analysis."""
    
    print("\n" + "="*70)
    print("教師:低品質 → 生徒:高品質 データでの蒸留分析")
    print("="*70)
    
    print("\n📊 状況:")
    print("-"*40)
    print("• 教師モデル: ノイジーな低品質データで学習")
    print("• 生徒モデル: クリーンな高品質データで蒸留")
    
    print("\n✅ なぜこれは良い状況か:")
    print("-"*40)
    print("1. 生徒は教師の「基礎知識」を獲得")
    print("2. 高品質GTデータが「正解」を提供")
    print("3. 教師のノイズは自然に除去される")
    print("4. 生徒は教師を超えられる！")
    
    print("\n🔄 学習プロセス:")
    print("-"*40)
    print("初期 (T=10, Epoch 1-30):")
    print("  • 教師から大まかなパターンを学習")
    print("  • ノイズは平滑化されて影響小")
    print("  • Loss = 0.7×KL + 0.3×BCE")
    print("")
    print("中期 (T=5, Epoch 30-70):")
    print("  • 高品質GTの影響が増加")
    print("  • 教師のノイズを修正開始")
    print("")
    print("後期 (T=1, Epoch 70-100):")
    print("  • GTデータが主導")
    print("  • 教師より高精度を達成")
    print("  • 教師の知識 + データの品質 = 最高の結果")
    
    print("\n💡 Temperature減衰の影響:")
    print("-"*40)
    print("後期で T→1 になっても問題なし:")
    print("• 教師の影響は強まるが...")
    print("• 高品質GTが常に正しい方向へ補正")
    print("• むしろ教師の有用な知識を確実に獲得")
    
    print("\n🎯 推奨設定:")
    print("-"*40)
    print("```python")
    print("# この状況では標準的な設定でOK")
    print("alpha = 0.7  # KL weight (教師の影響)")
    print("temperature_initial = 10")
    print("temperature_final = 1  # 問題なし")
    print("")
    print("# むしろGTを少し重視してもよい")
    print("alpha = 0.6  # KL weight を少し下げる")
    print("```")
    
    print("\n🚀 期待される結果:")
    print("-"*40)
    print("• 教師の性能: 82% (ノイズで制限)")
    print("• 生徒の性能: 90-95% (教師を超える！)")
    print("• Robustness: 教師より良い汎化性能")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Analyzing teacher (low quality) → student (high quality) case...")
    create_reverse_quality_visualization()
    print_detailed_analysis()
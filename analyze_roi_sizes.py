#!/usr/bin/env python3
"""
COCOデータセットのROIサイズ分析スクリプト

使用方法:
    # 検証用データセット（100枚）でテスト
    python analyze_roi_sizes.py --annotation data/annotations/instances_train2017_person_only_no_crowd_100.json
    
    # フルデータセットで分析
    python analyze_roi_sizes.py --annotation data/annotations/instances_train2017_person_only_no_crowd.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import seaborn as sns

def load_annotations(annotation_path: str) -> Dict:
    """COCOアノテーションファイルを読み込む"""
    with open(annotation_path, 'r') as f:
        return json.load(f)

def extract_roi_sizes(annotations: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アノテーションからROIサイズを抽出
    
    Returns:
        widths: ROIの幅の配列
        heights: ROIの高さの配列
        areas: ROIの面積の配列
    """
    widths = []
    heights = []
    areas = []
    
    for ann in annotations['annotations']:
        x, y, w, h = ann['bbox']
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
    
    return np.array(widths), np.array(heights), np.array(areas)

def calculate_statistics(data: np.ndarray, name: str) -> Dict:
    """統計情報を計算"""
    stats = {
        'name': name,
        'count': int(len(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'variance': float(np.var(data)),
        'median': float(np.median(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
        'p5': float(np.percentile(data, 5)),
        'p95': float(np.percentile(data, 95)),
    }
    return stats

def print_statistics(stats: Dict):
    """統計情報を見やすく表示"""
    print(f"\n=== {stats['name']} の統計情報 ===")
    print(f"サンプル数: {stats['count']:,}")
    print(f"平均値: {stats['mean']:.2f}")
    print(f"標準偏差: {stats['std']:.2f}")
    print(f"分散: {stats['variance']:.2f}")
    print(f"中央値: {stats['median']:.2f}")
    print(f"最小値: {stats['min']:.2f}")
    print(f"最大値: {stats['max']:.2f}")
    print(f"第1四分位数 (Q1): {stats['q1']:.2f}")
    print(f"第3四分位数 (Q3): {stats['q3']:.2f}")
    print(f"5パーセンタイル: {stats['p5']:.2f}")
    print(f"95パーセンタイル: {stats['p95']:.2f}")

def plot_distributions(widths: np.ndarray, heights: np.ndarray, areas: np.ndarray, 
                      output_dir: Path):
    """分布をプロット"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ROI Size Distribution Analysis', fontsize=16)
    
    # 幅の分布
    ax = axes[0, 0]
    ax.hist(widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.median(widths), color='red', linestyle='--', label=f'Median: {np.median(widths):.0f}')
    ax.axvline(np.mean(widths), color='green', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Width Distribution')
    ax.legend()
    
    # 高さの分布
    ax = axes[0, 1]
    ax.hist(heights, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.median(heights), color='red', linestyle='--', label=f'Median: {np.median(heights):.0f}')
    ax.axvline(np.mean(heights), color='blue', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
    ax.set_xlabel('Height (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Height Distribution')
    ax.legend()
    
    # 面積の分布
    ax = axes[0, 2]
    ax.hist(areas, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(np.median(areas), color='red', linestyle='--', label=f'Median: {np.median(areas):.0f}')
    ax.axvline(np.mean(areas), color='blue', linestyle='--', label=f'Mean: {np.mean(areas):.0f}')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Frequency')
    ax.set_title('Area Distribution')
    ax.legend()
    
    # ログスケールでの面積分布
    ax = axes[1, 0]
    ax.hist(areas, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Area Distribution (Log Scale)')
    
    # 幅と高さの散布図
    ax = axes[1, 1]
    ax.scatter(widths, heights, alpha=0.5, s=1)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.set_title('Width vs Height')
    
    # アスペクト比の分布
    ax = axes[1, 2]
    aspect_ratios = widths / heights
    ax.hist(aspect_ratios, bins=50, alpha=0.7, color='brown', edgecolor='black')
    ax.axvline(np.median(aspect_ratios), color='red', linestyle='--', 
               label=f'Median: {np.median(aspect_ratios):.2f}')
    ax.set_xlabel('Aspect Ratio (width/height)')
    ax.set_ylabel('Frequency')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roi_size_distribution.png', dpi=150)
    plt.close()
    
    # サイズ別の累積分布関数（CDF）をプロット
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cumulative Distribution Functions', fontsize=16)
    
    # 幅のCDF
    ax = axes[0]
    sorted_widths = np.sort(widths)
    cdf = np.arange(len(sorted_widths)) / float(len(sorted_widths))
    ax.plot(sorted_widths, cdf)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Width CDF')
    ax.grid(True)
    
    # 高さのCDF
    ax = axes[1]
    sorted_heights = np.sort(heights)
    cdf = np.arange(len(sorted_heights)) / float(len(sorted_heights))
    ax.plot(sorted_heights, cdf)
    ax.set_xlabel('Height (pixels)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Height CDF')
    ax.grid(True)
    
    # 面積のCDF
    ax = axes[2]
    sorted_areas = np.sort(areas)
    cdf = np.arange(len(sorted_areas)) / float(len(sorted_areas))
    ax.plot(sorted_areas, cdf)
    ax.set_xlabel('Area (pixels²)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Area CDF')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roi_size_cdf.png', dpi=150)
    plt.close()

def analyze_roi_scale_distribution(widths: np.ndarray, heights: np.ndarray) -> Dict:
    """
    ROIサイズの分布を分析し、最適なROIAlignとマスクサイズを提案
    """
    # 最大辺の長さを計算（width, heightの最大値）
    max_dims = np.maximum(widths, heights)
    
    # 異なるROIAlignサイズでのカバー率を計算
    roi_align_sizes = [14, 28, 56, 112]
    mask_sizes = [28, 56, 112, 224]
    
    analysis = {}
    
    for roi_size in roi_align_sizes:
        for mask_size in mask_sizes:
            if mask_size < roi_size:
                continue
                
            # ダウンサンプリング率
            downsample_ratio = max_dims / roi_size
            
            # 各ROIに対して、どの程度の詳細が保持されるかを評価
            # ダウンサンプリング率が1以下の場合、元のサイズ以上の解像度
            # ダウンサンプリング率が高いほど、詳細が失われる
            
            # 良好な品質の閾値（ダウンサンプリング率）
            excellent_threshold = 2.0  # 2倍以下のダウンサンプリング
            good_threshold = 4.0       # 4倍以下のダウンサンプリング
            acceptable_threshold = 8.0  # 8倍以下のダウンサンプリング
            
            excellent_coverage = np.sum(downsample_ratio <= excellent_threshold) / len(max_dims) * 100
            good_coverage = np.sum(downsample_ratio <= good_threshold) / len(max_dims) * 100
            acceptable_coverage = np.sum(downsample_ratio <= acceptable_threshold) / len(max_dims) * 100
            
            key = f"roi_{roi_size}_mask_{mask_size}"
            analysis[key] = {
                'roi_align_size': int(roi_size),
                'mask_size': int(mask_size),
                'excellent_coverage': float(excellent_coverage),
                'good_coverage': float(good_coverage),
                'acceptable_coverage': float(acceptable_coverage),
                'mean_downsample_ratio': float(np.mean(downsample_ratio)),
                'median_downsample_ratio': float(np.median(downsample_ratio)),
                'p95_downsample_ratio': float(np.percentile(downsample_ratio, 95)),
            }
    
    return analysis

def recommend_optimal_sizes(analysis: Dict) -> Tuple[int, int]:
    """
    分析結果から最適なROIAlignサイズとマスクサイズを推奨
    """
    # 評価基準:
    # 1. 80%以上のROIで「良好」な品質（4倍以下のダウンサンプリング）を維持
    # 2. 計算効率を考慮（できるだけ小さいサイズ）
    # 3. マスクサイズはROIAlignサイズの2倍程度が理想的（アップサンプリング品質）
    
    candidates = []
    
    for key, metrics in analysis.items():
        if metrics['good_coverage'] >= 80:  # 80%以上のROIで良好な品質
            # スコア計算（品質と効率のバランス）
            quality_score = (metrics['excellent_coverage'] * 2 + metrics['good_coverage']) / 3
            efficiency_score = 100 - (metrics['roi_align_size'] + metrics['mask_size']) / 3.36  # 正規化
            total_score = quality_score * 0.7 + efficiency_score * 0.3
            
            candidates.append({
                'roi_align_size': metrics['roi_align_size'],
                'mask_size': metrics['mask_size'],
                'score': total_score,
                'metrics': metrics
            })
    
    if candidates:
        # スコアが最も高い組み合わせを選択
        best = max(candidates, key=lambda x: x['score'])
        return best['roi_align_size'], best['mask_size'], best
    else:
        # デフォルト値
        return 56, 112, None

def main():
    parser = argparse.ArgumentParser(description='COCOデータセットのROIサイズ分析')
    parser.add_argument('--annotation', type=str, required=True,
                        help='COCOアノテーションファイルのパス')
    parser.add_argument('--output_dir', type=str, default='roi_analysis',
                        help='出力ディレクトリ')
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # アノテーションの読み込み
    print(f"アノテーションファイルを読み込み中: {args.annotation}")
    annotations = load_annotations(args.annotation)
    
    # ROIサイズの抽出
    widths, heights, areas = extract_roi_sizes(annotations)
    print(f"合計 {len(widths):,} 個のROIを分析")
    
    # 統計情報の計算と表示
    width_stats = calculate_statistics(widths, "幅（Width）")
    height_stats = calculate_statistics(heights, "高さ（Height）")
    area_stats = calculate_statistics(areas, "面積（Area）")
    
    print_statistics(width_stats)
    print_statistics(height_stats)
    print_statistics(area_stats)
    
    # アスペクト比の統計
    aspect_ratios = widths / heights
    aspect_stats = calculate_statistics(aspect_ratios, "アスペクト比（Width/Height）")
    print_statistics(aspect_stats)
    
    # 分布のプロット
    plot_distributions(widths, heights, areas, output_dir)
    
    # ROIスケール分析
    print("\n=== ROIスケール分析 ===")
    scale_analysis = analyze_roi_scale_distribution(widths, heights)
    
    # 分析結果を表形式で表示
    print("\n品質カバレッジ分析:")
    print("ROI Size | Mask Size | Excellent(<2x) | Good(<4x) | Acceptable(<8x) | Mean DS Ratio")
    print("-" * 80)
    
    for key in sorted(scale_analysis.keys()):
        metrics = scale_analysis[key]
        print(f"{metrics['roi_align_size']:8d} | {metrics['mask_size']:9d} | "
              f"{metrics['excellent_coverage']:14.1f}% | {metrics['good_coverage']:9.1f}% | "
              f"{metrics['acceptable_coverage']:15.1f}% | {metrics['mean_downsample_ratio']:13.2f}")
    
    # 最適なサイズの推奨
    optimal_roi, optimal_mask, best_candidate = recommend_optimal_sizes(scale_analysis)
    
    print(f"\n=== 推奨設定 ===")
    print(f"推奨ROIAlign出力サイズ: {optimal_roi}")
    print(f"推奨マスク出力サイズ: {optimal_mask}")
    
    if best_candidate:
        print(f"\n推奨理由:")
        print(f"- {best_candidate['metrics']['good_coverage']:.1f}%のROIで良好な品質（4倍以下のダウンサンプリング）")
        print(f"- {best_candidate['metrics']['excellent_coverage']:.1f}%のROIで優秀な品質（2倍以下のダウンサンプリング）")
        print(f"- 平均ダウンサンプリング率: {best_candidate['metrics']['mean_downsample_ratio']:.2f}")
        print(f"- 総合スコア: {best_candidate['score']:.2f}")
    
    # 現在の設定との比較
    current_roi = 28
    current_mask = 56
    current_metrics = scale_analysis.get(f"roi_{current_roi}_mask_{current_mask}")
    
    if current_metrics:
        print(f"\n=== 現在の設定（ROI: {current_roi}, Mask: {current_mask}）との比較 ===")
        print(f"現在の設定:")
        print(f"- 良好な品質カバレッジ: {current_metrics['good_coverage']:.1f}%")
        print(f"- 優秀な品質カバレッジ: {current_metrics['excellent_coverage']:.1f}%")
        print(f"- 平均ダウンサンプリング率: {current_metrics['mean_downsample_ratio']:.2f}")
        
        if optimal_roi != current_roi or optimal_mask != current_mask:
            print(f"\n改善点:")
            print(f"- 良好な品質カバレッジ: {current_metrics['good_coverage']:.1f}% → "
                  f"{best_candidate['metrics']['good_coverage']:.1f}% "
                  f"({best_candidate['metrics']['good_coverage'] - current_metrics['good_coverage']:+.1f}%)")
            print(f"- 優秀な品質カバレッジ: {current_metrics['excellent_coverage']:.1f}% → "
                  f"{best_candidate['metrics']['excellent_coverage']:.1f}% "
                  f"({best_candidate['metrics']['excellent_coverage'] - current_metrics['excellent_coverage']:+.1f}%)")
    
    # 結果をJSONで保存
    results = {
        'width_stats': width_stats,
        'height_stats': height_stats,
        'area_stats': area_stats,
        'aspect_ratio_stats': aspect_stats,
        'scale_analysis': scale_analysis,
        'recommendation': {
            'roi_align_size': optimal_roi,
            'mask_size': optimal_mask,
            'reasoning': best_candidate['metrics'] if best_candidate else None
        },
        'current_performance': current_metrics
    }
    
    with open(output_dir / 'roi_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n分析結果を {output_dir} に保存しました。")

if __name__ == "__main__":
    main()
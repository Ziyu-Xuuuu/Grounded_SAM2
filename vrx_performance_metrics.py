import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

class VRXPerformanceMetrics:
    """专门为VRX场景设计的性能指标"""
    
    def __init__(self):
        self.buoy_colors = {
            'red': [255, 0, 0],
            'green': [0, 255, 0], 
            'gray': [128, 128, 128],
            'black': [0, 0, 0],
            'orange': [255, 165, 0]
        }
    
    def calculate_stability_score(self, bbox_centers):
        """计算物体追踪的稳定性分数"""
        if len(bbox_centers) < 2:
            return 0.0
        
        centers = np.array(bbox_centers)
        # 计算相邻帧之间的位移
        displacements = np.diff(centers, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        
        # 稳定性分数：位移的标准差越小，稳定性越高
        stability = 1.0 / (1.0 + np.std(distances))
        return stability
    
    def calculate_size_consistency(self, mask_areas):
        """计算大小一致性"""
        if len(mask_areas) < 2:
            return 0.0
        
        areas = np.array([a for a in mask_areas if a > 0])
        if len(areas) < 2:
            return 0.0
        
        # 大小一致性：面积变化的变异系数
        consistency = 1.0 / (1.0 + np.std(areas) / np.mean(areas))
        return consistency
    
    def calculate_temporal_smoothness(self, confidence_scores):
        """计算时间连续性（置信度的平滑程度）"""
        if len(confidence_scores) < 3:
            return 0.0
        
        scores = np.array(confidence_scores)
        # 计算二阶差分（加速度）
        second_diff = np.diff(scores, n=2)
        smoothness = 1.0 / (1.0 + np.std(second_diff))
        return smoothness
    
    def plot_comprehensive_analysis(self, performance_data, object_id, buoy_type="unknown"):
        """综合性能分析图表"""
        fig = plt.figure(figsize=(20, 12))
        
        frames = performance_data['frames']
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 检测连续性热图
        ax1 = fig.add_subplot(gs[0, :2])
        detection_matrix = np.array(performance_data['mask_exists']).reshape(1, -1)
        im1 = ax1.imshow(detection_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        ax1.set_title(f'{buoy_type}浮标 (ID: {object_id}) - 检测连续性')
        ax1.set_xlabel('帧数')
        ax1.set_yticks([])
        
        # 2. 置信度和面积双轴图
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(frames, performance_data['confidence_scores'], 'b-', linewidth=2, label='置信度')
        line2 = ax2_twin.plot(frames, performance_data['mask_areas'], 'r-', linewidth=2, label='掩码面积')
        
        ax2.set_xlabel('帧数')
        ax2.set_ylabel('置信度分数', color='blue')
        ax2_twin.set_ylabel('掩码面积 (像素)', color='red')
        ax2.set_title('置信度与掩码面积关系')
        
        # 3. 位置轨迹 - 更详细
        ax3 = fig.add_subplot(gs[1, :2])
        centers = np.array(performance_data['bbox_centers'])
        if len(centers) > 0:
            # 彩色轨迹表示时间
            for i in range(len(centers)-1):
                alpha = i / len(centers)
                ax3.plot(centers[i:i+2, 0], centers[i:i+2, 1], 
                        color=plt.cm.viridis(alpha), linewidth=2, alpha=0.8)
            
            ax3.scatter(centers[0, 0], centers[0, 1], color='green', s=100, marker='o', label='起始点')
            ax3.scatter(centers[-1, 0], centers[-1, 1], color='red', s=100, marker='s', label='结束点')
            
        ax3.set_title('物体中心轨迹 (颜色表示时间)')
        ax3.set_xlabel('X 坐标')
        ax3.set_ylabel('Y 坐标')
        ax3.legend()
        ax3.invert_yaxis()
        
        # 4. 稳定性指标
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # 计算各种稳定性指标
        stability = self.calculate_stability_score(performance_data['bbox_centers'])
        size_consistency = self.calculate_size_consistency(performance_data['mask_areas'])
        temporal_smoothness = self.calculate_temporal_smoothness(performance_data['confidence_scores'])
        
        metrics = ['位置稳定性', '大小一致性', '时间平滑性']
        values = [stability, size_consistency, temporal_smoothness]
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylim(0, 1)
        ax4.set_title('追踪质量指标')
        ax4.set_ylabel('分数 (0-1)')
        
        # 在条形图上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 5. 丢失帧分析
        ax5 = fig.add_subplot(gs[2, :2])
        mask_exists = np.array(performance_data['mask_exists'])
        
        # 找到连续丢失的片段
        diff = np.diff(np.concatenate(([False], mask_exists, [False])).astype(int))
        lost_starts = np.where(diff == -1)[0]
        lost_ends = np.where(diff == 1)[0]
        
        ax5.plot(frames, mask_exists.astype(int), 'bo-', linewidth=2, markersize=4)
        
        # 标记丢失片段
        for start, end in zip(lost_starts, lost_ends):
            ax5.axvspan(frames[start], frames[end-1], alpha=0.3, color='red', label='丢失片段' if start == lost_starts[0] else "")
        
        ax5.set_title('检测状态时间线')
        ax5.set_xlabel('帧数')
        ax5.set_ylabel('是否检测到')
        ax5.set_ylim(-0.1, 1.1)
        if len(lost_starts) > 0:
            ax5.legend()
        
        # 6. 性能摘要
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        # 计算统计信息
        total_frames = len(frames)
        detected_frames = sum(performance_data['mask_exists'])
        detection_rate = detected_frames / total_frames * 100
        
        valid_confidences = [c for c, exists in zip(performance_data['confidence_scores'], performance_data['mask_exists']) if exists]
        avg_confidence = np.mean(valid_confidences) if valid_confidences else 0
        
        longest_gap = max([end - start for start, end in zip(lost_starts, lost_ends)]) if len(lost_starts) > 0 else 0
        
        summary_text = f"""
        === 性能摘要 ===
        浮标类型: {buoy_type}
        物体ID: {object_id}
        
        检测统计:
        • 总帧数: {total_frames}
        • 检测帧数: {detected_frames}
        • 检测率: {detection_rate:.1f}%
        • 最长丢失: {longest_gap} 帧
        
        质量指标:
        • 平均置信度: {avg_confidence:.3f}
        • 位置稳定性: {stability:.3f}
        • 大小一致性: {size_consistency:.3f}
        • 时间平滑性: {temporal_smoothness:.3f}
        
        总体评分: {np.mean([stability, size_consistency, temporal_smoothness]):.3f}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'VRX {buoy_type}浮标追踪性能综合分析 (ID: {object_id})', fontsize=16, fontweight='bold')
        
        return fig

# 使用示例
if __name__ == "__main__":
    # 这里需要与之前的 ObjectPerformanceAnalyzer 结合使用
    print("请先运行 plot_object_performance.py 获取 performance_data")
    print("然后使用 VRXPerformanceMetrics 进行更详细的分析")
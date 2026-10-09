"""
Visualization tools for deforestation detection results
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Dict


class ResultVisualizer:
    """Visualize deforestation detection results"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_prediction(self, image, prediction, save_path=None):
        """
        Plot image with prediction overlay
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction info
        status = "DEFORESTED" if prediction['is_deforested'] else "FORESTED"
        color = 'red' if prediction['is_deforested'] else 'green'
        
        axes[1].text(0.5, 0.7, status, 
                    ha='center', va='center', 
                    fontsize=24, fontweight='bold', 
                    color=color, transform=axes[1].transAxes)
        axes[1].text(0.5, 0.5, f"Confidence: {prediction['confidence']:.2%}", 
                    ha='center', va='center', 
                    fontsize=16, transform=axes[1].transAxes)
        axes[1].text(0.5, 0.3, f"Forest Coverage: {prediction['forest_coverage']:.2%}", 
                    ha='center', va='center', 
                    fontsize=16, transform=axes[1].transAxes)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.close()
    
    def plot_heatmap(self, image, heatmap, prediction, save_path=None):
        """
        Plot image with deforestation heatmap
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Deforestation Probability Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.6, 
                                 cv2.applyColorMap((heatmap * 255).astype(np.uint8), 
                                                  cv2.COLORMAP_JET), 
                                 0.4, 0)
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        plt.close()
    
    def plot_comparison(self, before_img, after_img, comparison_result, save_path=None):
        """
        Plot before/after comparison with change detection
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Before image
        axes[0].imshow(before_img)
        axes[0].set_title(f"Before\nCoverage: {comparison_result['before_coverage']:.2%}")
        axes[0].axis('off')
        
        # After image
        axes[1].imshow(after_img)
        axes[1].set_title(f"After\nCoverage: {comparison_result['after_coverage']:.2%}")
        axes[1].axis('off')
        
        # Change visualization
        diff = cv2.absdiff(before_img, after_img)
        axes[2].imshow(diff)
        axes[2].set_title(f"Change: {comparison_result['change_type'].upper()}\n"
                         f"Δ Coverage: {comparison_result['coverage_change']:.2%}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        plt.close()
    
    def plot_batch_results(self, results: List[Dict], images, save_path=None):
        """
        Plot batch prediction results in a grid
        """
        n_images = len(results)
        cols = 4
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx, (result, img) in enumerate(zip(results, images)):
            if idx >= len(axes):
                break
            
            axes[idx].imshow(img)
            status = "D" if result['is_deforested'] else "F"
            conf = result['confidence']
            color = 'red' if result['is_deforested'] else 'green'
            
            axes[idx].set_title(f"{status}: {conf:.2f}", color=color, fontweight='bold')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved batch results to {save_path}")
        
        plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history (loss and accuracy)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Forested', 'Deforested'],
                    yticklabels=['Forested', 'Deforested'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def create_report(self, results: List[Dict], save_path=None):
        """
        Create a summary report of predictions
        """
        total = len(results)
        deforested = sum(1 for r in results if r['is_deforested'])
        forested = total - deforested
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_coverage = np.mean([r['forest_coverage'] for r in results])
        
        report = f"""
Deforestation Detection Report
{'=' * 50}
Total Images Analyzed: {total}
  - Deforested: {deforested} ({deforested/total:.1%})
  - Forested: {forested} ({forested/total:.1%})

Average Confidence: {avg_confidence:.2%}
Average Forest Coverage: {avg_coverage:.2%}

Detailed Results:
{'-' * 50}
"""
        for i, result in enumerate(results, 1):
            status = "DEFORESTED" if result['is_deforested'] else "FORESTED"
            report += f"{i}. {Path(result['image_path']).name}\n"
            report += f"   Status: {status}\n"
            report += f"   Confidence: {result['confidence']:.2%}\n"
            report += f"   Forest Coverage: {result['forest_coverage']:.2%}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Saved report to {save_path}")
        
        return report


if __name__ == "__main__":
    # Test visualization
    visualizer = ResultVisualizer()
    
    # Create dummy data
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_prediction = {
        'is_deforested': True,
        'confidence': 0.85,
        'forest_coverage': 0.15,
        'image_path': 'test.png'
    }
    
    visualizer.plot_prediction(dummy_image, dummy_prediction, 
                               save_path='results/test_prediction.png')
    print("Visualization test complete!")

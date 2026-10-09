"""
Main application entry point for Deforestation Detection System
Provides CLI interface for training and inference
"""

import argparse
import sys
from pathlib import Path
import config
from train import train_model
from inference import DeforestationPredictor
from visualization import ResultVisualizer
from data_preprocessing import SatelliteImageProcessor


def train_command(args):
    """Handle training command"""
    print("\n" + "=" * 60)
    print("Training Deforestation Detection Model")
    print("=" * 60)
    
    # Override config if args provided
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.backbone:
        config.BACKBONE = args.backbone
    
    history, results = train_model()
    
    if args.plot_history:
        visualizer = ResultVisualizer()
        visualizer.plot_training_history(
            history, 
            save_path='results/training_history.png'
        )
    
    return 0


def predict_command(args):
    """Handle prediction command"""
    print("\n" + "=" * 60)
    print("Deforestation Detection Prediction")
    print("=" * 60)
    
    # Check if model exists
    model_path = args.model or 'models/deforestation_detector.h5'
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using: python main.py train")
        return 1
    
    # Initialize predictor
    predictor = DeforestationPredictor(model_path=model_path)
    visualizer = ResultVisualizer()
    
    if args.image:
        # Single image prediction
        print(f"\nProcessing single image: {args.image}")
        result = predictor.predict_single_image(args.image, threshold=args.threshold)
        
        # Print results
        print("\n" + "-" * 50)
        print(f"Image: {result['image_path']}")
        print(f"Status: {'DEFORESTED' if result['is_deforested'] else 'FORESTED'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Forest Coverage: {result['forest_coverage']:.2%}")
        print("-" * 50)
        
        # Visualization
        if args.visualize:
            processor = SatelliteImageProcessor()
            image = processor.load_image(args.image)
            if args.heatmap:
                heatmap_result = predictor.predict_with_heatmap(args.image, args.threshold)
                visualizer.plot_heatmap(
                    image, 
                    heatmap_result['heatmap_raw'],
                    result,
                    save_path='results/heatmap_prediction.png'
                )
            else:
                visualizer.plot_prediction(
                    image,
                    result,
                    save_path='results/prediction.png'
                )
    
    elif args.directory:
        # Batch prediction on directory
        print(f"\nProcessing directory: {args.directory}")
        results = predictor.predict_directory(args.directory, threshold=args.threshold, pattern=args.pattern)
        
        if not results:
            print("No images found or processed.")
            return 1
        
        # Print summary
        total = len(results)
        deforested = sum(1 for r in results if r['is_deforested'])
        print(f"\nProcessed {total} images")
        print(f"Deforested: {deforested} ({deforested/total:.1%})")
        print(f"Forested: {total - deforested} ({(total-deforested)/total:.1%})")
        
        # Generate report
        report = visualizer.create_report(results, save_path='results/detection_report.txt')
        print("\n" + report)
        
        # Visualization
        if args.visualize:
            processor = SatelliteImageProcessor()
            images = [processor.load_image(r['image_path']) for r in results if processor.load_image(r['image_path']) is not None]
            if images:
                visualizer.plot_batch_results(
                    results[:len(images)],
                    images,
                    save_path='results/batch_predictions.png'
                )
    
    elif args.compare:
        # Compare two images
        print(f"\nComparing images:")
        print(f"  Before: {args.compare[0]}")
        print(f"  After: {args.compare[1]}")
        
        comparison = predictor.compare_images(args.compare[0], args.compare[1])
        
        print("\n" + "-" * 50)
        print(f"Before Coverage: {comparison['before_coverage']:.2%}")
        print(f"After Coverage: {comparison['after_coverage']:.2%}")
        print(f"Coverage Change: {comparison['coverage_change']:.2%}")
        print(f"Change Type: {comparison['change_type'].upper()}")
        print("-" * 50)
        
        # Visualization
        if args.visualize:
            processor = SatelliteImageProcessor()
            before_img = processor.load_image(args.compare[0])
            after_img = processor.load_image(args.compare[1])
            if before_img is not None and after_img is not None:
                visualizer.plot_comparison(
                    before_img,
                    after_img,
                    comparison,
                    save_path='results/comparison.png'
                )
    
    else:
        print("Error: Please provide --image, --directory, or --compare")
        return 1
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Deforestation Detection System from Satellite Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --epochs 30 --batch-size 32

  # Predict on a single image
  python main.py predict --image path/to/image.png --visualize

  # Predict on a directory of images
  python main.py predict --directory path/to/images --visualize

  # Compare two images (before and after)
  python main.py predict --compare before.png after.png --visualize

  # Generate heatmap visualization
  python main.py predict --image path/to/image.png --heatmap --visualize
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the deforestation detection model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size for training')
    train_parser.add_argument('--backbone', choices=['resnet50', 'efficientnetb0', 'vgg16'],
                              help='Model backbone architecture')
    train_parser.add_argument('--plot-history', action='store_true',
                              help='Plot and save training history')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on satellite images')
    predict_parser.add_argument('--model', type=str, help='Path to trained model')
    predict_parser.add_argument('--image', type=str, help='Path to single image')
    predict_parser.add_argument('--directory', type=str, help='Path to directory of images')
    predict_parser.add_argument('--pattern', type=str, default='*.png',
                                help='File pattern for directory search (default: *.png)')
    predict_parser.add_argument('--compare', nargs=2, type=str,
                                help='Compare two images: before.png after.png')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                                help='Detection threshold (default: 0.5)')
    predict_parser.add_argument('--visualize', action='store_true',
                                help='Generate visualization of results')
    predict_parser.add_argument('--heatmap', action='store_true',
                                help='Generate heatmap visualization')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create necessary directories
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    Path(config.MODELS_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    
    # Execute command
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'predict':
        return predict_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

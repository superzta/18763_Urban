import time
import torch
from rcnn import Config, train, test, calculate_map

def compare_proxy_vs_exact():
    """Compare Proxy mAP vs Exact mAP (Performance & Value)"""
    print("\n" + "="*80)
    print("COMPARISON: Proxy mAP vs Exact mAP")
    print("="*80)
    
    # Use a small config for speed
    print("Training a small model for comparison...")
    config = Config(urban_issue_classes=[3])  # Single class
    config.num_epochs = 2
    config.batch_size = 4
    config.learning_rate = 0.005
    config.save_freq = 2
    
    # Train model once
    model = train(config)
    checkpoint_path = 'checkpoints/fasterrcnn_epoch_2.pth'
    
    print("\n" + "-"*40)
    print("Running Evaluation with PROXY mAP...")
    print("-" * 40)
    
    config.use_proxy_map = True
    start_time = time.time()
    proxy_results = test(config, checkpoint_path)
    proxy_time = time.time() - start_time
    
    print("\n" + "-"*40)
    print("Running Evaluation with EXACT mAP...")
    print("-" * 40)
    
    config.use_proxy_map = False
    start_time = time.time()
    exact_results = test(config, checkpoint_path)
    exact_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Metric':<20} | {'Proxy':<15} | {'Exact':<15} | {'Diff':<15}")
    print("-" * 75)
    
    map_proxy = proxy_results['mAP@0.5']
    map_exact = exact_results['mAP@0.5']
    print(f"{'mAP@0.5':<20} | {map_proxy:<15.4f} | {map_exact:<15.4f} | {map_proxy-map_exact:<15.4f}")
    
    prec_proxy = proxy_results['precision']
    prec_exact = exact_results['precision']
    print(f"{'Precision':<20} | {prec_proxy:<15.4f} | {prec_exact:<15.4f} | {prec_proxy-prec_exact:<15.4f}")
    
    rec_proxy = proxy_results['recall']
    rec_exact = exact_results['recall']
    print(f"{'Recall':<20} | {rec_proxy:<15.4f} | {rec_exact:<15.4f} | {rec_proxy-rec_exact:<15.4f}")
    
    print(f"{'Time (sec)':<20} | {proxy_time:<15.4f} | {exact_time:<15.4f} | {exact_time-proxy_time:<15.4f}")
    print("-" * 75)
    print("Note: Exact mAP is usually lower than Proxy mAP because Proxy assumes a perfect PR curve.")


def sanity_check_map():
    """
    Quick synthetic test to verify calculate_map() behavior
    without training or loading any model.

    We construct:
      - 1 image
      - 2 GT boxes of class 1
      - 3 predictions:
          p1: high score, matches GT1  -> TP
          p2: medium score, matches GT2 -> TP
          p3: low score, far away       -> FP

    This gives:
      - total GT = 2
      - total preds = 3 (2 TP, 1 FP)
    """

    # 1 image ground truth
    gt_boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],  # GT1
        [60.0, 60.0, 100.0, 100.0] # GT2
    ], dtype=torch.float32)
    gt_labels = torch.tensor([1, 1], dtype=torch.int64)

    ground_truths = [{
        "boxes": gt_boxes,
        "labels": gt_labels,
        "image_id": torch.tensor([0]),
        "area": (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 2] - gt_boxes[:, 0]),
        "iscrowd": torch.zeros(len(gt_boxes), dtype=torch.int64),
    }]

    # 3 predictions for class 1
    pred_boxes = torch.tensor([
        [12.0, 12.0, 48.0, 48.0],   # p1: IoU ~1 with GT1 -> TP
        [62.0, 62.0, 98.0, 98.0],   # p2: IoU ~1 with GT2 -> TP
        [150.0, 150.0, 200.0, 200.0]# p3: no overlap -> FP
    ], dtype=torch.float32)
    pred_labels = torch.tensor([1, 1, 1], dtype=torch.int64)
    pred_scores = torch.tensor([0.9, 0.7, 0.3], dtype=torch.float32)

    predictions = [{
        "boxes": pred_boxes,
        "labels": pred_labels,
        "scores": pred_scores,
    }]

    print("\n" + "="*80)
    print("SANITY CHECK: calculate_map on toy data")
    print("="*80)

    # Proxy mAP with a threshold that keeps only high-scoring detections
    proxy_results = calculate_map(
        predictions,
        ground_truths,
        conf_threshold=0.5,
        iou_threshold=0.5,
        use_proxy=True,
    )

    # Exact mAP: uses all detections, ignores conf_threshold internally
    exact_results = calculate_map(
        predictions,
        ground_truths,
        conf_threshold=0.5,
        iou_threshold=0.5,
        use_proxy=False,
    )

    print("\nToy Ground Truths: 2 boxes of class 1")
    print("Toy Predictions: 3 boxes of class 1 (2 TP, 1 FP)")
    print("- Proxy mAP uses conf_threshold=0.5 (keeps 2 TPs, drops 1 FP)")
    print("- Exact mAP uses all 3 predictions and builds full PR curve\n")

    print(f"Proxy mAP@0.5: {proxy_results['mAP@0.5']:.4f}")
    print(f"Exact  mAP@0.5: {exact_results['mAP@0.5']:.4f}")
    print(f"Proxy precision: {proxy_results['precision']:.4f}, recall: {proxy_results['recall']:.4f}")
    print(f"Exact  precision: {exact_results['precision']:.4f}, recall: {exact_results['recall']:.4f}")
    print(f"Total GTs: {proxy_results['total_ground_truths']}, "
          f"Total preds: {proxy_results['total_predictions']}")
    print("="*80 + "\n")


if __name__ == '__main__':
    compare_proxy_vs_exact()
    # sanity_check_map()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import lightgbm as lgb
import joblib
import os

def evaluate_model(test_csv, model_dir='../models', output_dir='../results'):
    """Evaluate trained model on test set"""
    
    # Load test data
    df = pd.read_csv(test_csv)
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    
    X_test = df[feature_cols]
    y_test = df['label']
    
    # Load model
    model = lgb.Booster(model_file=os.path.join(model_dir, 'lightgbm_model.txt'))
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Evaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'AI-Generated']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (Real correctly identified): {cm[0,0]}")
    print(f"False Positives (Real misclassified as AI): {cm[0,1]}")
    print(f"False Negatives (AI misclassified as Real): {cm[1,0]}")
    print(f"True Positives (AI correctly identified): {cm[1,1]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # Plot Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-Generated'],
                yticklabels=['Real', 'AI-Generated'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance
    importance_df = pd.read_csv(os.path.join(model_dir, 'feature_importance.csv'))
    
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    sns.barplot(data=top_features, y='feature', x='importance')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Precision at different thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    precisions_at_thresh = []
    recalls_at_thresh = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba > thresh).astype(int)
        if y_pred_thresh.sum() > 0:  # Avoid division by zero
            prec = precision_score(y_test, y_pred_thresh)
            rec = recall_score(y_test, y_pred_thresh)
        else:
            prec, rec = 0, 0
        precisions_at_thresh.append(prec)
        recalls_at_thresh.append(rec)
    
    print(f"\nPrecision at different thresholds:")
    for i, thresh in enumerate(thresholds):
        print(f"Threshold {thresh:.1f}: Precision = {precisions_at_thresh[i]:.4f}, Recall = {recalls_at_thresh[i]:.4f}")
    
    # Save detailed results
    results = {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'thresholds': thresholds.tolist(),
        'precisions_at_thresh': precisions_at_thresh,
        'recalls_at_thresh': recalls_at_thresh
    }
    
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <test_features.csv> [model_dir] [output_dir]")
        sys.exit(1)
    
    test_csv = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else '../models'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '../results'
    
    results = evaluate_model(test_csv, model_dir, output_dir)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

def train_meta_classifier(features_csv, model_output_dir='../models'):
    """Train the meta-classifier using extracted features"""
    
    # Load data
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} samples")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['label', 'image_path', 'filename', 'source']]
    X = df[feature_cols]
    y = df['label']
    
    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])
    
    print(f"Features: {len(feature_cols)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'verbosity': -1,
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20
    }
    
    # Create datasets
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_test, y_test, reference=dtrain)
    
    # Train model
    print("Training LightGBM model...")
    bst = lgb.train(
        params, 
        dtrain, 
        num_boost_round=500,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'eval'],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Predictions
    train_preds = bst.predict(X_train)
    test_preds = bst.predict(X_test)
    
    # Evaluation
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print(f"\nTraining AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Classification report
    test_pred_binary = (test_preds > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred_binary))
    
    # Feature importance
    importance = bst.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_imp.head(10))
    
    # Save model and metadata
    os.makedirs(model_output_dir, exist_ok=True)
    
    model_path = os.path.join(model_output_dir, 'lightgbm_model.txt')
    bst.save_model(model_path)
    
    # Save feature names and importance
    feature_imp.to_csv(os.path.join(model_output_dir, 'feature_importance.csv'), index=False)
    
    # Save feature columns for inference
    joblib.dump(feature_cols, os.path.join(model_output_dir, 'feature_columns.pkl'))
    
    print(f"\nModel saved to: {model_path}")
    
    return bst, feature_cols, test_auc

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <features.csv>")
        sys.exit(1)
    
    features_csv = sys.argv[1]
    model, features, auc = train_meta_classifier(features_csv)

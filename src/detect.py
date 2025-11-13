import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
try:
    from .feature_extract import extract_all_features
except ImportError:
    from feature_extract import extract_all_features

class AIImageDetector:
    def __init__(self, model_dir='../models'):
        self.model_dir = model_dir
        self.model = None
        self.feature_columns = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and feature columns"""
        model_path = os.path.join(self.model_dir, 'lightgbm_model.txt')
        feature_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature columns not found at {feature_path}")
        
        self.model = lgb.Booster(model_file=model_path)
        self.feature_columns = joblib.load(feature_path)
        print(f"Loaded model with {len(self.feature_columns)} features")
    
    def predict_single(self, image_path, return_features=False):
        """Predict if a single image is AI-generated"""
        try:
            # Check image size first
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            min_dimension = min(width, height)
            
            # Extract features
            features = extract_all_features(image_path)
            
            # Create feature vector in correct order
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Predict - keep original probability
            prob = self.model.predict(feature_vector)[0]
            
            # Only adjust confidence display, not the actual prediction
            if min_dimension < 256:
                confidence_penalty = 0.8
            elif min_dimension < 512:
                confidence_penalty = 0.9
            else:
                confidence_penalty = 1.0
            
            # Calculate confidence (don't modify the actual prediction)
            base_confidence = float(max(prob, 1-prob))
            adjusted_confidence = base_confidence * confidence_penalty
            
            result = {
                'image_path': image_path,
                'ai_probability': float(prob),  # Keep original probability
                'prediction': 1 if prob > 0.5 else 0,  # Use original threshold
                'confidence': adjusted_confidence,
                'image_size': f"{width}x{height}",
                'size_warning': min_dimension < 512
            }
            
            if return_features:
                result['features'] = features
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'ai_probability': None,
                'prediction': 'Error',
                'confidence': 0.0
            }
    
    def predict_batch(self, image_paths):
        """Predict for multiple images"""
        results = []
        for path in image_paths:
            result = self.predict_single(path)
            results.append(result)
        return results
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        importance_path = os.path.join(self.model_dir, 'feature_importance.csv')
        if os.path.exists(importance_path):
            return pd.read_csv(importance_path)
        else:
            importance = self.model.feature_importance(importance_type='gain')
            return pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path> [model_dir]")
        print("       python detect.py <directory> [model_dir]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else '../models'
    
    # Initialize detector
    detector = AIImageDetector(model_dir)
    
    # Check if input is file or directory
    if os.path.isfile(input_path):
        # Single image
        result = detector.predict_single(input_path, return_features=True)
        
        print(f"\nImage: {result['image_path']}")
        print(f"Prediction: {result['prediction']}")
        print(f"AI Probability: {result['ai_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    elif os.path.isdir(input_path):
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_paths = []
        
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(input_path, filename))
        
        if not image_paths:
            print("No image files found in directory")
            sys.exit(1)
        
        print(f"Processing {len(image_paths)} images...")
        results = detector.predict_batch(image_paths)
        
        # Summary
        ai_count = sum(1 for r in results if r.get('ai_probability', 0) > 0.5)
        real_count = len(results) - ai_count
        
        print(f"\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"AI-Generated: {ai_count}")
        print(f"Real: {real_count}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for result in results:
            filename = os.path.basename(result['image_path'])
            prob = result.get('ai_probability', 0)
            pred = result.get('prediction', 'Error')
            print(f"{filename:30} | {pred:12} | {prob:.4f}")
    
    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()

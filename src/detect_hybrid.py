import os
import sys
import numpy as np
import joblib
import lightgbm as lgb

# Import feature extraction
try:
    from .feature_extract import extract_all_features
    from .metadata_display import extract_metadata_for_display
except ImportError:
    from feature_extract import extract_all_features
    from metadata_display import extract_metadata_for_display

class HybridAIImageDetector:
    def __init__(self, binary_model_dir='models', three_class_model_path='three_class_model.pkl'):
        """Initialize with both binary and three-class models"""
        self.binary_model_dir = binary_model_dir
        self.three_class_model_path = three_class_model_path
        self.load_models()
    
    def load_models(self):
        """Load both binary and three-class models"""
        # Load binary model
        model_path = os.path.join(self.binary_model_dir, 'lightgbm_model.txt')
        feature_path = os.path.join(self.binary_model_dir, 'feature_columns.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Binary model not found at {model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature columns not found at {feature_path}")
        
        self.binary_model = lgb.Booster(model_file=model_path)
        self.feature_columns = joblib.load(feature_path)
        
        # Load three-class model
        if not os.path.exists(self.three_class_model_path):
            raise FileNotFoundError(f"Three-class model not found at {self.three_class_model_path}")
        
        self.three_class_model = joblib.load(self.three_class_model_path)
        
        print(f"✅ Loaded binary model with {len(self.feature_columns)} features")
        print(f"✅ Loaded three-class model")
    
    def predict_single(self, image_path, return_features=False):
        """Predict using hybrid approach with metadata override"""
        try:
            # Extract features
            features = extract_all_features(image_path)
            
            # Extract metadata
            metadata = extract_metadata_for_display(image_path)
            camera_info = metadata.get('camera', {})
            has_camera_metadata = 'Make' in camera_info or 'Model' in camera_info
            
            # Get image size
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            
            # METADATA OVERRIDE - If camera metadata exists, classify as Real
            if has_camera_metadata:
                result = {
                    'image_path': image_path,
                    'prediction': 0,  # Real
                    'ai_probability': 0.05,
                    'confidence': 0.95,
                    'three_class_prediction': 'Real Photo (Camera Detected)',
                    'binary_model_prediction': None,  # Will fill below
                    'three_class_model_prediction': None,  # Will fill below
                    'metadata_override': True,
                    'camera_info': camera_info,
                    'image_size': f"{width}x{height}",
                    'size_warning': min(width, height) < 512
                }
            else:
                # No metadata - use model predictions
                result = {
                    'image_path': image_path,
                    'metadata_override': False,
                    'camera_info': {},
                    'image_size': f"{width}x{height}",
                    'size_warning': min(width, height) < 512
                }
            
            # Get binary model prediction (for comparison)
            feature_vector_binary = []
            for col in self.feature_columns:
                feature_vector_binary.append(features.get(col, 0.0))
            feature_vector_binary = np.array(feature_vector_binary).reshape(1, -1)
            
            binary_prob = self.binary_model.predict(feature_vector_binary)[0]
            binary_prediction = 1 if binary_prob > 0.5 else 0
            
            # Get three-class model prediction
            feature_vector_3class = np.array(list(features.values())).reshape(1, -1)
            class_pred = self.three_class_model.predict(feature_vector_3class)[0]
            class_proba = self.three_class_model.predict_proba(feature_vector_3class)[0]
            
            class_names = ['Real Photo', 'AI Generated', 'Digital Content']
            three_class_prediction = class_names[class_pred]
            
            # Store model predictions for analysis
            result['binary_model_prediction'] = {
                'prediction': binary_prediction,
                'ai_probability': float(binary_prob),
                'classification': 'AI' if binary_prediction == 1 else 'Real'
            }
            
            result['three_class_model_prediction'] = {
                'prediction': class_pred,
                'classification': three_class_prediction,
                'probabilities': {
                    'Real Photo': float(class_proba[0]),
                    'AI Generated': float(class_proba[1]),
                    'Digital Content': float(class_proba[2])
                }
            }
            
            # If no metadata override, use BINARY MODEL as primary (it's more reliable)
            if not result['metadata_override']:
                # Use binary model as primary decision
                result['prediction'] = binary_prediction
                result['ai_probability'] = float(binary_prob)
                result['confidence'] = float(max(binary_prob, 1-binary_prob))
                
                # Add three-class info for additional context
                if binary_prediction == 1:
                    result['three_class_prediction'] = 'AI Generated (Binary Model)'
                else:
                    # Use three-class model to distinguish Real vs Digital for Real predictions
                    if class_pred == 0:
                        result['three_class_prediction'] = 'Real Photo'
                    else:
                        result['three_class_prediction'] = 'Digital Content'
            
            if return_features:
                result['features'] = features
                result['metadata'] = metadata
            
            return result
            
        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_hybrid.py <image_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Initialize hybrid detector
    detector = HybridAIImageDetector()
    
    if os.path.isfile(input_path):
        result = detector.predict_single(input_path, return_features=True)
        
        print(f"\nImage: {result['image_path']}")
        print(f"Final Prediction: {'AI-Generated' if result['prediction'] == 1 else 'Real Photo'}")
        print(f"Three-Class: {result['three_class_prediction']}")
        print(f"AI Probability: {result['ai_probability']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if result['metadata_override']:
            print(f"📷 METADATA OVERRIDE APPLIED")
            make = result['camera_info'].get('Make', 'Unknown')
            model = result['camera_info'].get('Model', 'Unknown')
            print(f"Camera: {make} {model}")
        else:
            print(f"\nModel Predictions:")
            print(f"  Binary Model: {result['binary_model_prediction']['classification']} (prob: {result['binary_model_prediction']['ai_probability']:.3f})")
            print(f"  Three-Class Model: {result['three_class_model_prediction']['classification']}")
    else:
        print(f"File not found: {input_path}")

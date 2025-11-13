import streamlit as st
import sys
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# Make HEIF support optional
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

sys.path.append('src')

st.set_page_config(
    page_title="AI Image Detector - Visual Analysis", 
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_detector():
    try:
        from src.detect_hybrid import HybridAIImageDetector
        return HybridAIImageDetector('models', 'three_class_model.pkl')
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None

detector = load_detector()

st.title("📊 AI Image Detector - Visual Analysis")
st.write("Comprehensive visual analysis with charts, distributions, and quality metrics")

uploaded_file = st.file_uploader(
    "Upload image for visual analysis", 
    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'] + (['heic'] if HEIF_SUPPORTED else [])
)

if uploaded_file and detector:
    try:
        # Process image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            result = detector.predict_single(tmp_file.name, return_features=True)
            
            # Load image for additional analysis
            img = cv2.imread(tmp_file.name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.open(tmp_file.name)
        
        # Main Results Dashboard
        st.subheader("🎯 Detection Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Classification", result['three_class_prediction'])
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with col3:
            st.metric("AI Probability", f"{result['ai_probability']:.1%}")
        with col4:
            st.metric("Image Size", result.get('image_size', 'Unknown'))
        
        # Visual Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📷 Image Analysis", 
            "🎨 Color Distribution", 
            "📊 Feature Insights", 
            "🔍 Quality Metrics",
            "📋 Metadata Summary"
        ])
        
        with tab1:  # Image Analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                st.image(pil_img, use_column_width=True)
                
                # Image properties
                width, height = pil_img.size
                st.write(f"**Dimensions:** {width} × {height}")
                st.write(f"**Aspect Ratio:** {width/height:.2f}")
                st.write(f"**File Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")
            
            with col2:
                st.subheader("RGB Histogram Analysis")
                
                # Create histogram data
                hist_data = []
                colors = ['Red', 'Green', 'Blue']
                for i in range(3):
                    channel_data = img_rgb[:,:,i].flatten()
                    hist_data.append(pd.DataFrame({
                        'Value': channel_data,
                        'Channel': colors[i]
                    }))
                
                combined_hist = pd.concat(hist_data)
                st.bar_chart(combined_hist.groupby(['Channel']).size())
        
        with tab2:  # Color Distribution
            st.subheader("🎨 Color Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Color statistics from features
                features = result.get('features', {})
                color_stats = []
                
                for channel in ['r', 'g', 'b']:
                    mean_key = f'{channel}_mean'
                    std_key = f'{channel}_std'
                    if mean_key in features and std_key in features:
                        color_stats.append({
                            'Channel': channel.upper(),
                            'Mean': features[mean_key],
                            'Std Dev': features[std_key]
                        })
                
                if color_stats:
                    df_colors = pd.DataFrame(color_stats)
                    st.subheader("Average Color Values")
                    st.bar_chart(df_colors.set_index('Channel')['Mean'])
            
            with col2:
                # Brightness and contrast metrics
                brightness = features.get('brightness', 0)
                contrast = features.get('contrast', 0)
                
                st.subheader("Image Quality Metrics")
                st.metric("Brightness", f"{brightness * 255:.1f}/255")
                st.metric("Contrast", f"{contrast:.3f}")
                
                # Progress bars for visual representation
                st.write("**Brightness Level:**")
                st.progress(brightness)
                st.write("**Contrast Level:**")
                st.progress(min(contrast, 1.0))
        
        with tab3:  # Feature Insights
            st.subheader("📊 Feature Analysis")
            
            features = result.get('features', {})
            
            # Group features by importance/category
            feature_groups = {
                'Spectral Features': [k for k in features.keys() if 'fft' in k or 'freq' in k][:6],  # Top 6
                'Statistical Features': [k for k in features.keys() if any(x in k for x in ['mean', 'std', 'var'])],
                'Metadata Features': [k for k in features.keys() if any(x in k for x in ['software', 'exif', 'metadata'])],
                'Quality Features': [k for k in features.keys() if any(x in k for x in ['noise', 'compression', 'quality'])]
            }
            
            for group_name, feature_list in feature_groups.items():
                if feature_list:
                    st.write(f"**{group_name}:**")
                    
                    # Create DataFrame for visualization
                    group_data = []
                    for feat in feature_list[:8]:  # Limit to top 8 per group
                        if feat in features:
                            group_data.append({
                                'Feature': feat.replace('_', ' ').title(),
                                'Value': float(features[feat]) if isinstance(features[feat], (int, float)) else 0
                            })
                    
                    if group_data:
                        df_group = pd.DataFrame(group_data)
                        st.bar_chart(df_group.set_index('Feature')['Value'])
        
        with tab4:  # Quality Metrics
            st.subheader("🔍 Image Quality Assessment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Quality metrics
                quality_metrics = {}
                
                # Extract quality-related features
                for key, value in features.items():
                    if any(term in key.lower() for term in ['noise', 'blur', 'compression', 'quality', 'sharpness']):
                        quality_metrics[key.replace('_', ' ').title()] = value
                
                if quality_metrics:
                    st.write("**Quality Metrics:**")
                    for metric, value in quality_metrics.items():
                        st.metric(metric, f"{value:.4f}")
            
            with col2:
                # Size and format analysis
                st.write("**Technical Specifications:**")
                
                # Create metrics display
                tech_specs = {
                    "Resolution": f"{width} × {height}",
                    "Megapixels": f"{(width * height) / 1_000_000:.1f} MP",
                    "File Format": uploaded_file.type,
                    "Color Mode": pil_img.mode,
                    "Has Transparency": "Yes" if pil_img.mode in ['RGBA', 'LA'] else "No"
                }
                
                for spec, value in tech_specs.items():
                    st.write(f"• **{spec}:** {value}")
                
                # Size warning
                if result.get('size_warning'):
                    st.warning("⚠️ Image resolution is below optimal size (512px)")
        
        with tab5:  # Metadata Summary
            st.subheader("📋 Metadata Overview")
            
            metadata = result.get('metadata', {})
            
            if metadata:
                # Metadata categories with counts
                categories = ['camera', 'technical', 'location', 'other']
                category_data = []
                
                for category in categories:
                    cat_data = metadata.get(category, {})
                    category_data.append({
                        'Category': category.title(),
                        'Fields Found': len(cat_data),
                        'Has Data': len(cat_data) > 0
                    })
                
                df_meta = pd.DataFrame(category_data)
                
                # Bar chart for metadata availability
                st.subheader("Metadata Availability by Category")
                st.bar_chart(df_meta.set_index('Category')['Fields Found'])
                
                # Show detailed metadata
                for category in categories:
                    cat_data = metadata.get(category, {})
                    if cat_data:
                        st.write(f"**{category.title()} Information:**")
                        for key, value in cat_data.items():
                            st.write(f"• **{key}:** {value}")
                        st.write("---")
                
                # Camera detection highlight
                if result.get('metadata_override'):
                    st.success("✅ **Camera Detected** - Authentic photo metadata found")
                    camera_info = result.get('camera_info', {})
                    if camera_info:
                        st.write("**Camera Details:**")
                        for key, value in camera_info.items():
                            st.write(f"• **{key}:** {value}")
                else:
                    st.info("📱 No camera metadata detected - relying on AI analysis")
            else:
                st.warning("No metadata found in this image")
        
        # Model Comparison
        if 'binary_model_prediction' in result and 'three_class_model_prediction' in result:
            st.subheader("🤖 Model Predictions Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Binary model results
                binary_pred = result['binary_model_prediction']
                st.write("**Binary Model (AI vs Real):**")
                st.metric("AI Probability", f"{binary_pred['ai_probability']:.1%}")
                st.metric("Classification", binary_pred['classification'])
            
            with col2:
                # Three-class model results
                three_pred = result['three_class_model_prediction']
                st.write("**Three-Class Model Probabilities:**")
                
                if 'probabilities' in three_pred:
                    probs = three_pred['probabilities']
                    prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
                    prob_df['Probability'] = prob_df['Probability'] * 100
                    
                    st.bar_chart(prob_df.set_index('Class')['Probability'])
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

elif not detector:
    st.error("❌ AI detector not loaded. Please check model files.")
else:
    st.info("👆 Upload an image to start visual analysis")

# Add requirements note
st.markdown("---")
st.markdown("**📊 Visual Analysis Features:** Built-in Streamlit charts, distributions, quality metrics, and comprehensive insights")

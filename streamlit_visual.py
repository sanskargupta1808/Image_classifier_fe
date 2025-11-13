import streamlit as st
import sys
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
                st.subheader("Histogram Analysis")
                
                # RGB Histograms
                fig = go.Figure()
                
                for i, color in enumerate(['Red', 'Green', 'Blue']):
                    hist, bins = np.histogram(img_rgb[:,:,i].flatten(), bins=50)
                    fig.add_trace(go.Scatter(
                        x=bins[:-1], y=hist,
                        mode='lines', name=color,
                        line=dict(color=['red', 'green', 'blue'][i])
                    ))
                
                fig.update_layout(
                    title="RGB Color Distribution",
                    xaxis_title="Pixel Value",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
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
                    
                    # Bar chart for means
                    fig_mean = px.bar(df_colors, x='Channel', y='Mean', 
                                    title='Average Color Values',
                                    color='Channel',
                                    color_discrete_map={'R': 'red', 'G': 'green', 'B': 'blue'})
                    st.plotly_chart(fig_mean, use_container_width=True)
            
            with col2:
                # Brightness and contrast metrics
                brightness = features.get('brightness', 0)
                contrast = features.get('contrast', 0)
                
                # Gauge charts
                fig_brightness = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = brightness * 255,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Brightness"},
                    gauge = {
                        'axis': {'range': [None, 255]},
                        'bar': {'color': "yellow"},
                        'steps': [
                            {'range': [0, 85], 'color': "lightgray"},
                            {'range': [85, 170], 'color': "gray"},
                            {'range': [170, 255], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 200
                        }
                    }
                ))
                fig_brightness.update_layout(height=300)
                st.plotly_chart(fig_brightness, use_container_width=True)
        
        with tab3:  # Feature Insights
            st.subheader("📊 Feature Analysis")
            
            features = result.get('features', {})
            
            # Group features by importance/category
            feature_groups = {
                'Spectral Features': [k for k in features.keys() if 'fft' in k or 'freq' in k],
                'Statistical Features': [k for k in features.keys() if any(x in k for x in ['mean', 'std', 'var'])],
                'Metadata Features': [k for k in features.keys() if any(x in k for x in ['software', 'exif', 'metadata'])],
                'Quality Features': [k for k in features.keys() if any(x in k for x in ['noise', 'compression', 'quality'])]
            }
            
            for group_name, feature_list in feature_groups.items():
                if feature_list:
                    st.write(f"**{group_name}:**")
                    
                    # Create DataFrame for visualization
                    group_data = []
                    for feat in feature_list[:10]:  # Limit to top 10 per group
                        if feat in features:
                            group_data.append({
                                'Feature': feat.replace('_', ' ').title(),
                                'Value': float(features[feat]) if isinstance(features[feat], (int, float)) else 0
                            })
                    
                    if group_data:
                        df_group = pd.DataFrame(group_data)
                        
                        # Horizontal bar chart
                        fig = px.bar(df_group, x='Value', y='Feature', 
                                   orientation='h',
                                   title=f'{group_name} Distribution')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
        
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
                    # Radar chart for quality metrics
                    categories = list(quality_metrics.keys())
                    values = list(quality_metrics.values())
                    
                    # Normalize values for radar chart
                    max_val = max(abs(v) for v in values) if values else 1
                    normalized_values = [v/max_val for v in values]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=categories,
                        fill='toself',
                        name='Quality Metrics'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[-1, 1]
                            )),
                        showlegend=False,
                        title="Quality Assessment Radar"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Size and format analysis
                st.write("**Technical Specifications:**")
                
                # Create metrics display
                tech_metrics = [
                    ("Resolution", f"{width} × {height}"),
                    ("Megapixels", f"{(width * height) / 1_000_000:.1f} MP"),
                    ("File Format", uploaded_file.type),
                    ("Color Mode", pil_img.mode),
                    ("Has Transparency", "Yes" if pil_img.mode in ['RGBA', 'LA'] else "No")
                ]
                
                for metric, value in tech_metrics:
                    st.write(f"• **{metric}:** {value}")
                
                # Size warning
                if result.get('size_warning'):
                    st.warning("⚠️ Image resolution is below optimal size (512px)")
        
        with tab5:  # Metadata Summary
            st.subheader("📋 Metadata Overview")
            
            metadata = result.get('metadata', {})
            
            if metadata:
                # Metadata categories with visual indicators
                categories = ['camera', 'technical', 'location', 'other']
                category_data = []
                
                for category in categories:
                    cat_data = metadata.get(category, {})
                    category_data.append({
                        'Category': category.title(),
                        'Fields Found': len(cat_data),
                        'Has Data': 'Yes' if cat_data else 'No'
                    })
                
                df_meta = pd.DataFrame(category_data)
                
                # Bar chart for metadata availability
                fig = px.bar(df_meta, x='Category', y='Fields Found',
                           title='Metadata Availability by Category',
                           color='Has Data',
                           color_discrete_map={'Yes': 'green', 'No': 'red'})
                st.plotly_chart(fig, use_container_width=True)
                
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
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = binary_pred['ai_probability'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AI Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "red" if binary_pred['ai_probability'] > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Three-class model results
                three_pred = result['three_class_model_prediction']
                st.write("**Three-Class Model Probabilities:**")
                
                probs = three_pred['probabilities']
                prob_df = pd.DataFrame(list(probs.items()), columns=['Class', 'Probability'])
                prob_df['Probability'] = prob_df['Probability'] * 100
                
                fig = px.pie(prob_df, values='Probability', names='Class',
                           title='Classification Probabilities')
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

elif not detector:
    st.error("❌ AI detector not loaded. Please check model files.")
else:
    st.info("👆 Upload an image to start visual analysis")

# Add requirements note
st.markdown("---")
st.markdown("**📊 Visual Analysis Features:** Charts, distributions, quality metrics, and comprehensive insights")

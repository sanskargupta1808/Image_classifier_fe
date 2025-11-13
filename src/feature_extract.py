import io
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import piexif
import math
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.restoration import denoise_wavelet
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from pillow_heif import register_heif_opener
import numpy as np

# Register HEIF opener for HEIC support
register_heif_opener()

def load_image(path, target_size=None):
    """Load and preprocess image"""
    im = Image.open(path).convert("RGB")
    if target_size:
        im.thumbnail(target_size, Image.LANCZOS)
    return np.array(im)

def extract_exif_features(path):
    """Extract metadata features from image"""
    features = {
        'software_len': 0, 'suspicious_software': 0, 'has_exif': 0, 
        'has_make': 0, 'has_model': 0, 'has_gps': 0, 'metadata_count': 0,
        'metadata_richness': 0, 'has_camera_settings': 0
    }
    
    try:
        # Try PIL first (works with HEIC)
        img = Image.open(path)
        exif = img.getexif()
        
        if exif:
            features['has_exif'] = 1
            features['metadata_count'] = len(exif)
            
            # Check for make (271) and model (272)
            make = exif.get(271, '')
            model = exif.get(272, '')
            software = exif.get(305, '')  # Software tag
            
            # Camera settings indicators
            camera_tags = [
                34665,  # EXIF IFD
                34853,  # GPS IFD
                37386,  # Focal length
                33434,  # Exposure time
                33437,  # F-number
                34855,  # ISO
                37377,  # Shutter speed
                37378,  # Aperture
                37380,  # Exposure bias
                37383,  # Metering mode
                37384,  # Light source
                37385,  # Flash
            ]
            
            camera_settings_count = sum(1 for tag in camera_tags if tag in exif)
            features['has_camera_settings'] = 1 if camera_settings_count > 3 else 0
            
            # GPS data check
            features['has_gps'] = 1 if 34853 in exif else 0
            
            # Metadata richness score (0-100)
            richness_score = min(100, (len(exif) * 5) + (camera_settings_count * 10))
            features['metadata_richness'] = richness_score
            
            if make:
                features['has_make'] = 1
            if model:
                features['has_model'] = 1
            
            # Analyze software tag
            if software:
                features['software_len'] = len(str(software))
                
                # Check for AI/editing software keywords
                software_lower = str(software).lower()
                ai_keywords = ['midjourney', 'stable', 'sd', 'dalle', 'gpt', 'photoshop', 'synth', 'ai', 'generated', 'artificial']
                features['suspicious_software'] = 1 if any(kw in software_lower for kw in ai_keywords) else 0
        
    except Exception as e:
        # Fallback to piexif for JPEG/TIFF
        try:
            exif_dict = piexif.load(path)
            
            # Count total metadata entries
            total_entries = sum(len(ifd_dict) for ifd_dict in exif_dict.values() if isinstance(ifd_dict, dict))
            features['metadata_count'] = total_entries
            features['has_exif'] = 1 if total_entries > 0 else 0
            
            # GPS data
            features['has_gps'] = 1 if exif_dict.get('GPS') else 0
            
            # Camera settings
            exif_ifd = exif_dict.get('Exif', {})
            features['has_camera_settings'] = 1 if len(exif_ifd) > 5 else 0
            
            # Metadata richness
            features['metadata_richness'] = min(100, total_entries * 3)
            
            # Software tag analysis
            software = exif_dict['0th'].get(piexif.ImageIFD.Software, b'').decode('utf-8', errors='ignore')
            make = exif_dict['0th'].get(piexif.ImageIFD.Make, b'').decode('utf-8', errors='ignore')
            model = exif_dict['0th'].get(piexif.ImageIFD.Model, b'').decode('utf-8', errors='ignore')
            
            features['software_len'] = len(software)
            features['has_make'] = 1 if make else 0
            features['has_model'] = 1 if model else 0
            
            # Check for AI/editing software keywords
            ai_keywords = ['midjourney', 'stable', 'sd', 'dalle', 'gpt', 'photoshop', 'synth', 'ai', 'generated', 'artificial']
            features['suspicious_software'] = 1 if any(kw in software.lower() for kw in ai_keywords) else 0
            
        except Exception:
            pass  # No EXIF data available
    
    return features

def fft_features(img_rgb, num_radial_bins=24):
    """Extract frequency domain features"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = img_as_float(gray)
    
    # 2D FFT
    F = fftshift(fft2(gray))
    mag = np.abs(F)
    mag_log = np.log1p(mag)
    
    h, w = mag_log.shape
    cy, cx = h//2, w//2
    
    # Radial binning
    max_r = math.hypot(cx, cy)
    bin_edges = np.linspace(0, max_r, num_radial_bins+1)
    yy, xx = np.indices(mag_log.shape)
    r = np.hypot(xx-cx, yy-cy)
    
    radial_means = []
    for i in range(num_radial_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if mask.sum() == 0:
            radial_means.append(0.0)
        else:
            radial_means.append(mag_log[mask].mean())
    
    radial_means = np.array(radial_means)
    
    # High/low frequency ratios
    hf = radial_means[num_radial_bins//2:].mean()
    lf = radial_means[:num_radial_bins//2].mean()
    hf_lf_ratio = (hf / (lf + 1e-9))
    
    features = {f'fft_radial_{i}': float(radial_means[i]) for i in range(num_radial_bins)}
    features.update({
        'fft_hf': float(hf),
        'fft_lf': float(lf),
        'fft_hf_lf_ratio': float(hf_lf_ratio),
        'fft_radial_std': float(radial_means.std()),
        'fft_radial_mean': float(radial_means.mean())
    })
    
    return features

def noise_features(img_rgb):
    """Extract PRNU/noise residual features"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    
    # Wavelet denoising (updated API)
    denoised = denoise_wavelet(gray, rescale_sigma=True)
    residual = gray - denoised
    
    # Statistical features
    features = {
        'resid_mean': float(residual.mean()),
        'resid_std': float(residual.std()),
        'resid_skew': float(np.mean((residual - residual.mean())**3) / (residual.std()**3 + 1e-9)),
        'resid_kurt': float(np.mean((residual - residual.mean())**4) / (residual.std()**4 + 1e-9))
    }
    
    # FFT of residual
    F = fftshift(fft2(residual))
    mag = np.abs(F)
    mag_log = np.log1p(mag)
    features['resid_fft_mean'] = float(mag_log.mean())
    features['resid_fft_std'] = float(mag_log.std())
    
    return features

def gan_fingerprint_features(img_rgb):
    """Detect GAN-specific artifacts and fingerprints"""
    features = {}
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Checkerboard artifacts (common in GANs)
    kernel_checkerboard = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    checkerboard_response = cv2.filter2D(gray, -1, kernel_checkerboard)
    features['checkerboard_artifacts'] = float(np.std(checkerboard_response))
    
    # 2. Upsampling artifacts detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    features['gradient_regularity'] = float(np.std(gradient_mag))
    
    # 3. Local Binary Pattern analysis
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    features['texture_uniformity'] = float(np.std(lbp))
    
    # 4. High-frequency noise patterns
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['hf_noise_pattern'] = float(np.mean(np.abs(laplacian)))
    
    return features

def structural_forensic_features(img_rgb):
    """Extract structural and forensic features"""
    features = {}
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Edge consistency analysis
    edges = cv2.Canny(gray, 50, 150)
    features['edge_density'] = float(np.sum(edges > 0) / edges.size)
    
    # Lighting consistency
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    lighting_angles = np.arctan2(grad_y, grad_x + 1e-8)
    features['lighting_consistency'] = float(np.std(lighting_angles))
    
    # Color channel correlations
    r, g, b = img_rgb[:,:,0].flatten(), img_rgb[:,:,1].flatten(), img_rgb[:,:,2].flatten()
    features['color_correlation_rg'] = float(np.corrcoef(r, g)[0, 1])
    features['color_correlation_rb'] = float(np.corrcoef(r, b)[0, 1])
    
    return features

def c2pa_content_credentials(path):
    """Check for C2PA/Content Credentials"""
    features = {'has_c2pa': 0, 'content_credentials': 0}
    
    try:
        img = Image.open(path)
        exif = img.getexif()
        
        c2pa_indicators = ['c2pa', 'content credentials', 'provenance', 'cai']
        for tag_id, value in exif.items():
            if isinstance(value, (str, bytes)):
                if any(indicator in str(value).lower() for indicator in c2pa_indicators):
                    features['has_c2pa'] = 1
                    features['content_credentials'] = 1
                    break
    except Exception:
        pass
    
    return features

def estimate_jpeg_quality(path):
    """Estimate JPEG quality (placeholder implementation)"""
    return {'jpeg_quality_est': -1}

def extract_all_features(path, target_size=(512,512)):
    """Extract all features from an image (excluding unreliable EXIF metadata)"""
    img = load_image(path, target_size)
    
    features = {}
    # Remove EXIF features - unreliable due to editing/cropping
    # features.update(extract_exif_features(path))
    features.update(fft_features(img, num_radial_bins=24))
    features.update(noise_features(img))
    features.update(estimate_jpeg_quality(path))
    features.update(gan_fingerprint_features(img))
    features.update(structural_forensic_features(img))
    # Remove C2PA as it's not widely adopted yet
    # features.update(c2pa_content_credentials(path))
    
    # Basic image properties
    features['width'] = int(img.shape[1])
    features['height'] = int(img.shape[0])
    features['aspect_ratio'] = float(img.shape[1]) / (img.shape[0] + 1e-9)
    
    return features

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python feature_extract.py <image_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    features = extract_all_features(path)
    print(json.dumps(features, indent=2))

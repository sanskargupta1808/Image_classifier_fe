import os
import pandas as pd
from feature_extract import extract_all_features
from tqdm import tqdm
import argparse

def prepare_dataset(real_images_dir, ai_images_dir, output_csv):
    """Prepare dataset by extracting features from real and AI images"""
    
    data = []
    
    # Process real images
    if os.path.exists(real_images_dir):
        print("Processing real images...")
        real_files = [f for f in os.listdir(real_images_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        for filename in tqdm(real_files, desc="Real images"):
            try:
                path = os.path.join(real_images_dir, filename)
                features = extract_all_features(path)
                features['label'] = 0  # Real
                features['image_path'] = path
                features['filename'] = filename
                data.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Process AI images
    if os.path.exists(ai_images_dir):
        print("Processing AI-generated images...")
        ai_files = [f for f in os.listdir(ai_images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
        
        for filename in tqdm(ai_files, desc="AI images"):
            try:
                path = os.path.join(ai_images_dir, filename)
                features = extract_all_features(path)
                features['label'] = 1  # AI
                features['image_path'] = path
                features['filename'] = filename
                data.append(features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"\nDataset prepared:")
    print(f"Total samples: {len(df)}")
    print(f"Real images: {sum(df['label'] == 0)}")
    print(f"AI images: {sum(df['label'] == 1)}")
    print(f"Features: {len(df.columns) - 3}")  # Exclude label, image_path, filename
    print(f"Saved to: {output_csv}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for AI image detection')
    parser.add_argument('--real-dir', required=True, help='Directory containing real images')
    parser.add_argument('--ai-dir', required=True, help='Directory containing AI-generated images')
    parser.add_argument('--output', default='../data/features.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Prepare dataset
    df = prepare_dataset(args.real_dir, args.ai_dir, args.output)

if __name__ == "__main__":
    main()

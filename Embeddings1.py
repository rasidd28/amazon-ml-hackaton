import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
import requests
from io import BytesIO
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AMAZON ML CHALLENGE - MULTIMODAL FEATURE EXTRACTION")
print("=" * 80)

# ====================================================================================
# SECTION 1: ENVIRONMENT CONFIGURATION
# ====================================================================================
print("\n[INITIALIZATION] Configuring environment parameters...")

# Directory structure
PROJECT_ROOT = "/home/me22b1026/amazon_ml_hackathon"
TRAINING_FILE = f"{PROJECT_ROOT}/dataset/train.csv"
TESTING_FILE = f"{PROJECT_ROOT}/dataset/test.csv"
IMAGE_CACHE = f"{PROJECT_ROOT}/images"
FEATURE_STORAGE = f"{PROJECT_ROOT}/embeddings"

# Initialize folders
os.makedirs(IMAGE_CACHE, exist_ok=True)
os.makedirs(FEATURE_STORAGE, exist_ok=True)

# Processing parameters
PROCESSING_BATCH = 32
PARALLEL_THREADS = 64
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"✓ Project root: {PROJECT_ROOT}")
print(f"✓ Image cache location: {IMAGE_CACHE}")
print(f"✓ Feature storage location: {FEATURE_STORAGE}")
print(f"✓ Computing device: {COMPUTE_DEVICE}")
print(f"✓ CUDA availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")

# Load datasets
print("\n[DATA LOADING] Reading CSV files...")
training_dataset = pd.read_csv(TRAINING_FILE)
testing_dataset = pd.read_csv(TESTING_FILE)
print(f"✓ Training records: {len(training_dataset):,}")
print(f"✓ Testing records: {len(testing_dataset):,}")

# ====================================================================================
# SECTION 2: IMAGE RETRIEVAL UTILITIES
# ====================================================================================
print("\n" + "=" * 80)
print("[IMAGE RETRIEVAL] Setting up download mechanisms")
print("=" * 80)

def fetch_image_from_url(image_url, destination_path, timeout_sec=5, retry_count=2):
    """Retrieve image with retry logic and exception handling"""
    if os.path.exists(destination_path):
        return True, "cached"
    
    for attempt_num in range(retry_count):
        try:
            http_response = requests.get(image_url, timeout=timeout_sec)
            if http_response.status_code == 200:
                image_obj = Image.open(BytesIO(http_response.content)).convert("RGB")
                image_obj.save(destination_path)
                return True, "fetched"
        except Exception as error:
            if attempt_num == retry_count - 1:
                return False, str(error)
            time.sleep(0.5)
    return False, "timeout"

def construct_image_path(url_string, record_id):
    """Build standardized filename from URL or record identifier"""
    try:
        extracted_name = url_string.split('/')[-1].split('?')[0]
        if not extracted_name.endswith(('.jpg', '.jpeg', '.png')):
            extracted_name = f"{record_id}.jpg"
    except:
        extracted_name = f"{record_id}.jpg"
    return extracted_name

def retrieve_dataset_images(dataframe, dataset_label="train"):
    """Batch download images with parallel execution"""
    print(f"\n{'=' * 80}")
    print(f"Retrieving {dataset_label.upper()} dataset images...")
    print(f"{'=' * 80}")
    
    outcome_stats = {"fetched": 0, "error": 0, "cached": 0}
    error_records = []
    
    # Build download queue
    download_queue = []
    for row_idx, data_row in dataframe.iterrows():
        img_url = data_row['image_link']
        record_id = data_row['sample_id']
        file_name = construct_image_path(img_url, record_id)
        full_path = os.path.join(IMAGE_CACHE, file_name)
        download_queue.append((img_url, full_path, record_id))
    
    # Execute parallel downloads
    with ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
        job_mapping = {executor.submit(fetch_image_from_url, url, path): (rec_id, path) 
                      for url, path, rec_id in download_queue}
        
        with tqdm(total=len(job_mapping), desc=f"{dataset_label} images", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as progress:
            for completed_job in as_completed(job_mapping):
                rec_id, path = job_mapping[completed_job]
                success_flag, result_status = completed_job.result()
                
                if success_flag:
                    if result_status == "cached":
                        outcome_stats["cached"] += 1
                    else:
                        outcome_stats["fetched"] += 1
                else:
                    outcome_stats["error"] += 1
                    error_records.append(rec_id)
                
                progress.update(1)
                progress.set_postfix({
                    'Downloaded': outcome_stats["fetched"],
                    'Cached': outcome_stats["cached"],
                    'Errors': outcome_stats["error"]
                })
    
    # Display results
    print(f"\n{dataset_label.upper()} Retrieval Summary:")
    print(f"  ✓ Newly downloaded: {outcome_stats['fetched']:,}")
    print(f"  ✓ Already cached: {outcome_stats['cached']:,}")
    print(f"  ✗ Failed downloads: {outcome_stats['error']:,}")
    
    if error_records:
        print(f"  ⚠ Failed IDs (sample): {error_records[:10]}")
    
    return error_records

# ====================================================================================
# SECTION 3: EXECUTE IMAGE DOWNLOADS
# ====================================================================================
print("\n" + "=" * 80)
print("[BATCH DOWNLOAD] Fetching all product images")
print("=" * 80)

# Retrieve training images
training_failures = retrieve_dataset_images(training_dataset, "train")

# Retrieve testing images
testing_failures = retrieve_dataset_images(testing_dataset, "test")

cached_count = len([f for f in os.listdir(IMAGE_CACHE) if f.endswith(('.jpg', '.jpeg', '.png'))])
print(f"\n✅ Image retrieval complete!")
print(f"Total cached images: {cached_count:,}")

# ====================================================================================
# SECTION 4: INITIALIZE ENCODING MODELS
# ====================================================================================
print("\n" + "=" * 80)
print("[MODEL LOADING] Initializing neural encoders")
print("=" * 80)

print("\n⏳ Loading text encoder (BAAI/bge-large-en-v1.5)...")
text_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=COMPUTE_DEVICE)
print("✓ Text encoder initialized")

print("\n⏳ Loading vision encoder (CLIP-ViT-Large-Patch14)...")
vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(COMPUTE_DEVICE)
vision_preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("✓ Vision encoder initialized")

print(f"\n📊 Encoder specifications:")
print(f"  Text feature dimension: 1024")
print(f"  Vision feature dimension: 768")
print(f"  Concatenated dimension: 1792")

# ====================================================================================
# SECTION 5: TEXT FEATURE EXTRACTION
# ====================================================================================
print("\n" + "=" * 80)
print("[TEXT ENCODING] Extracting semantic text features")
print("=" * 80)

def extract_text_features(dataframe, processing_batch=128):
    """Generate semantic embeddings from product descriptions"""
    print(f"\nEncoding {len(dataframe):,} text samples...")
    
    # Prepare text corpus
    text_corpus = dataframe['catalog_content'].fillna("").astype(str).tolist()
    
    # Batch encoding
    feature_matrix = text_encoder.encode(
        text_corpus,
        batch_size=processing_batch,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"✓ Text feature matrix: {feature_matrix.shape}")
    return feature_matrix

# Extract training text features
train_text_file = os.path.join(FEATURE_STORAGE, "train_text.npy")

if os.path.exists(train_text_file):
    print("\n✓ Loading cached TRAINING text features...")
    train_text_features = np.load(train_text_file)
    print(f"  Dimensions: {train_text_features.shape}")
else:
    print("\n→ Extracting TRAINING text features...")
    train_text_features = extract_text_features(training_dataset)
    np.save(train_text_file, train_text_features)
    print(f"✓ Saved to: {train_text_file}")

# Extract testing text features
test_text_file = os.path.join(FEATURE_STORAGE, "test_text.npy")

if os.path.exists(test_text_file):
    print("\n✓ Loading cached TESTING text features...")
    test_text_features = np.load(test_text_file)
    print(f"  Dimensions: {test_text_features.shape}")
else:
    print("\n→ Extracting TESTING text features...")
    test_text_features = extract_text_features(testing_dataset)
    np.save(test_text_file, test_text_features)
    print(f"✓ Saved to: {test_text_file}")

# ====================================================================================
# SECTION 6: VISION FEATURE EXTRACTION
# ====================================================================================
print("\n" + "=" * 80)
print("[VISION ENCODING] Extracting visual image features")
print("=" * 80)

def extract_vision_features(dataframe, processing_batch=32):
    """Generate visual embeddings using CLIP encoder"""
    print(f"\nEncoding {len(dataframe):,} image samples...")
    
    feature_collection = []
    failed_samples = []
    
    # Build image path list
    image_paths = []
    for row_idx, data_row in dataframe.iterrows():
        file_name = construct_image_path(data_row['image_link'], data_row['sample_id'])
        full_path = os.path.join(IMAGE_CACHE, file_name)
        image_paths.append((row_idx, full_path))
    
    # Process in batches
    with tqdm(total=len(image_paths), desc="Vision encoding",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as progress:
        
        for batch_start in range(0, len(image_paths), processing_batch):
            batch_subset = image_paths[batch_start:batch_start+processing_batch]
            batch_images = []
            batch_indices = []
            
            # Load batch images
            for row_idx, img_path in batch_subset:
                try:
                    if os.path.exists(img_path):
                        img_data = Image.open(img_path).convert("RGB")
                        batch_images.append(img_data)
                        batch_indices.append(row_idx)
                    else:
                        failed_samples.append(row_idx)
                        feature_collection.append(np.zeros(768))
                except Exception as load_error:
                    failed_samples.append(row_idx)
                    feature_collection.append(np.zeros(768))
            
            # Encode batch
            if batch_images:
                processed_inputs = vision_preprocessor(
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(COMPUTE_DEVICE)
                
                with torch.no_grad():
                    visual_features = vision_encoder.get_image_features(**processed_inputs)
                    visual_features = visual_features.cpu().numpy()
                
                feature_collection.extend(visual_features)
            
            progress.update(len(batch_subset))
            if failed_samples:
                progress.set_postfix({'Failures': len(failed_samples)})
    
    feature_array = np.array(feature_collection)
    print(f"\n✓ Vision feature matrix: {feature_array.shape}")
    
    if failed_samples:
        print(f"⚠ Failed samples: {len(failed_samples):,} (zero vectors used)")
    
    return feature_array

# Extract training vision features
train_vision_file = os.path.join(FEATURE_STORAGE, "train_img.npy")

if os.path.exists(train_vision_file):
    print("\n✓ Loading cached TRAINING vision features...")
    train_vision_features = np.load(train_vision_file)
    print(f"  Dimensions: {train_vision_features.shape}")
else:
    print("\n→ Extracting TRAINING vision features...")
    train_vision_features = extract_vision_features(training_dataset, PROCESSING_BATCH)
    np.save(train_vision_file, train_vision_features)
    print(f"✓ Saved to: {train_vision_file}")

# Extract testing vision features
test_vision_file = os.path.join(FEATURE_STORAGE, "test_img.npy")

if os.path.exists(test_vision_file):
    print("\n✓ Loading cached TESTING vision features...")
    test_vision_features = np.load(test_vision_file)
    print(f"  Dimensions: {test_vision_features.shape}")
else:
    print("\n→ Extracting TESTING vision features...")
    test_vision_features = extract_vision_features(testing_dataset, PROCESSING_BATCH)
    np.save(test_vision_file, test_vision_features)
    print(f"✓ Saved to: {test_vision_file}")

# ====================================================================================
# SECTION 7: COMPLETION SUMMARY
# ====================================================================================
print("\n" + "=" * 80)
print("[SUMMARY] Feature extraction completed")
print("=" * 80)

print(f"""
Generated Features:
  
  TRAINING SET:
    • Text features: {train_text_features.shape} → [{train_text_file}]
    • Vision features: {train_vision_features.shape} → [{train_vision_file}]
  
  TESTING SET:
    • Text features: {test_text_features.shape} → [{test_text_file}]
    • Vision features: {test_vision_features.shape} → [{test_vision_file}]

Storage Information:
  • Feature directory: {FEATURE_STORAGE}
  • Training text: {os.path.getsize(train_text_file) / 1024**2:.2f} MB
  • Training vision: {os.path.getsize(train_vision_file) / 1024**2:.2f} MB
  • Testing text: {os.path.getsize(test_text_file) / 1024**2:.2f} MB
  • Testing vision: {os.path.getsize(test_vision_file) / 1024**2:.2f} MB

Next Phase:
  ✓ All features cached and ready for reuse
  → Execute 3_training.ipynb for model development
""")

print("\n✅ FEATURE EXTRACTION PIPELINE COMPLETED!")
print("=" * 80)

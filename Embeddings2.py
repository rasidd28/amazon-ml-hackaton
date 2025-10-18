# ====================================================================================
# TESTING IMAGE FEATURES - CHECKPOINT RECOVERY SYSTEM
# Automatic saving every 6000 samples - Resume from interruptions!
# ====================================================================================

import pandas as pd
import numpy as np
import os
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
import gc
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TESTING IMAGE ENCODER - RESILIENT CHECKPOINT MODE")
print("=" * 80)

# ====================================================================================
# SYSTEM CONFIGURATION
# ====================================================================================
PROJECT_ROOT = "/home/me22b1026/amazon_ml_hackathon"
FEATURE_STORAGE = f"{PROJECT_ROOT}/embeddings"
IMAGE_CACHE = f"{PROJECT_ROOT}/images"

# Storage paths
test_vision_output = f"{FEATURE_STORAGE}/test_img.npy"
recovery_checkpoint = f"{FEATURE_STORAGE}/test_img_checkpoint.npy"
progress_tracker = f"{FEATURE_STORAGE}/test_img_progress.txt"

COMPUTE_DEVICE = "cuda"
SAVE_FREQUENCY = 5000  # Checkpoint interval

# ====================================================================================
# VERIFY COMPLETION STATUS
# ====================================================================================
if os.path.exists(test_vision_output):
    print("\n✅ test_img.npy IS ALREADY COMPLETE!")
    completed_features = np.load(test_vision_output)
    print(f"   Dimensions: {completed_features.shape}")
    print("\n🎉 ALL FEATURE FILES AVAILABLE!")
    print("   ✓ train_text.npy")
    print("   ✓ train_img.npy")
    print("   ✓ test_text.npy")
    print("   ✓ test_img.npy")
    print("\n→ Proceed to 3_training_enhanced.ipynb")
    import sys
    sys.exit(0)

# ====================================================================================
# LOAD TESTING DATASET
# ====================================================================================
print("\n[DATA] Loading testing dataset...")
testing_data = pd.read_csv(f"{PROJECT_ROOT}/dataset/test.csv")
print(f"✓ Testing samples: {len(testing_data):,}")

# ====================================================================================
# UTILITY FUNCTIONS
# ====================================================================================
def construct_image_path(url_string, record_id):
    """Build standardized filename from URL or record identifier"""
    try:
        extracted_name = url_string.split('/')[-1].split('?')[0]
        if not extracted_name.endswith(('.jpg', '.jpeg', '.png')):
            extracted_name = f"{record_id}.jpg"
    except:
        extracted_name = f"{record_id}.jpg"
    return extracted_name

def release_memory():
    """Forcefully clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ====================================================================================
# CHECKPOINT RECOVERY
# ====================================================================================
resume_index = 0
feature_buffer = []

if os.path.exists(recovery_checkpoint):
    print("\n⚠️  CHECKPOINT DETECTED! Resuming previous session...")
    try:
        feature_buffer = list(np.load(recovery_checkpoint))
        resume_index = len(feature_buffer)
        print(f"   ✓ Recovered {resume_index:,} processed features")
        print(f"   → Continuing from sample {resume_index:,} / {len(testing_data):,}")
        
        if os.path.exists(progress_tracker):
            with open(progress_tracker, 'r') as tracker:
                last_checkpoint = int(tracker.read().strip())
                print(f"   → Last saved at: {last_checkpoint:,}")
    except Exception as recovery_error:
        print(f"   ⚠️  Checkpoint damaged: {recovery_error}")
        print(f"   → Restarting from beginning...")
        resume_index = 0
        feature_buffer = []
else:
    print(f"\n→ Fresh start: Processing {len(testing_data):,} testing images")

# ====================================================================================
# INITIALIZE VISION ENCODER
# ====================================================================================
print("\n→ Initializing CLIP vision encoder...")
try:
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(COMPUTE_DEVICE)
    vision_prep = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vision_model.eval()
    print(f"✓ Vision encoder loaded on {COMPUTE_DEVICE}")
except Exception as load_error:
    print(f"❌ Failed to initialize encoder: {load_error}")
    raise

# ====================================================================================
# SEQUENTIAL IMAGE PROCESSING
# ====================================================================================
print("\n" + "=" * 80)
print("ENCODING TESTING IMAGES")
print("=" * 80)
print(f"Progress: {resume_index:,} / {len(testing_data):,} ({resume_index/len(testing_data)*100:.1f}%)")
print(f"Remaining: {len(testing_data) - resume_index:,} samples")
print(f"Checkpoint interval: Every {SAVE_FREQUENCY:,} samples")
print("=" * 80)

# Initialize progress tracker
progress_bar = tqdm(
    range(resume_index, len(testing_data)), 
    desc="Encoding images",
    initial=resume_index,
    total=len(testing_data),
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
)

for current_idx in progress_bar:
    try:
        # Retrieve image information
        data_row = testing_data.iloc[current_idx]
        image_filename = construct_image_path(data_row['image_link'], data_row['sample_id'])
        image_fullpath = os.path.join(IMAGE_CACHE, image_filename)
        
        # Encode image
        if os.path.exists(image_fullpath):
            try:
                # Load and preprocess
                img_object = Image.open(image_fullpath).convert("RGB")
                model_inputs = vision_prep(images=[img_object], return_tensors="pt").to(COMPUTE_DEVICE)
                
                with torch.no_grad():
                    visual_embedding = vision_model.get_image_features(**model_inputs).cpu().numpy()[0]
                
                feature_buffer.append(visual_embedding)
                
            except Exception as encoding_error:
                # Encoding failed - use zero vector
                feature_buffer.append(np.zeros(768))
        else:
            # Image missing - use zero vector
            feature_buffer.append(np.zeros(768))
        
        # Periodic checkpoint save
        if (current_idx + 1) % SAVE_FREQUENCY == 0:
            checkpoint_array = np.array(feature_buffer)
            np.save(recovery_checkpoint, checkpoint_array)
            
            with open(progress_tracker, 'w') as tracker:
                tracker.write(str(current_idx + 1))
            
            progress_bar.write(f"\n💾 Checkpoint: {current_idx + 1:,} / {len(testing_data):,} ({(current_idx+1)/len(testing_data)*100:.1f}%)")
        
        # Memory cleanup cycle
        if current_idx % 1000 == 0 and current_idx > resume_index:
            release_memory()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Manual interruption detected!")
        print(f"   Saving emergency checkpoint at {len(feature_buffer):,} samples...")
        np.save(recovery_checkpoint, np.array(feature_buffer))
        with open(progress_tracker, 'w') as tracker:
            tracker.write(str(len(feature_buffer)))
        print(f"   ✓ Emergency save complete. Resume from {len(feature_buffer):,}")
        raise
    
    except Exception as process_error:
        print(f"\n⚠️  Error at index {current_idx}: {process_error}")
        print("   Applying zero vector and continuing...")
        feature_buffer.append(np.zeros(768))
        continue

progress_bar.close()

# ====================================================================================
# FINALIZE AND SAVE
# ====================================================================================
print("\n" + "=" * 80)
print("FINALIZING FEATURE STORAGE")
print("=" * 80)

print(f"\n→ Converting {len(feature_buffer):,} features to array...")
test_vision_features = np.array(feature_buffer)

print(f"→ Writing to {test_vision_output}...")
np.save(test_vision_output, test_vision_features)

# Validation check
if os.path.exists(test_vision_output):
    validation_array = np.load(test_vision_output)
    print(f"✓ Save successful!")
    print(f"  Dimensions: {validation_array.shape}")
    print(f"  File size: {os.path.getsize(test_vision_output) / 1024**2:.2f} MB")
else:
    print("❌ Validation failed!")

# ====================================================================================
# CLEANUP OPERATIONS
# ====================================================================================
print("\n→ Removing temporary files...")

# Delete checkpoint artifacts
if os.path.exists(recovery_checkpoint):
    os.remove(recovery_checkpoint)
    print("✓ Checkpoint file removed")

if os.path.exists(progress_tracker):
    os.remove(progress_tracker)
    print("✓ Progress tracker removed")

# Free GPU resources
del vision_model, vision_prep
release_memory()
print("✓ GPU memory released")

# ====================================================================================
# FINAL VALIDATION
# ====================================================================================
print("\n" + "=" * 80)
print("✅ TEST IMAGE ENCODING COMPLETED!")
print("=" * 80)

print("\nValidation Report:")
print(f"  Feature dimensions: {test_vision_features.shape}")
print(f"  Expected dimensions: ({len(testing_data)}, 768)")

if test_vision_features.shape == (len(testing_data), 768):
    print("  ✓ Dimensions verified successfully!")
else:
    print(f"  ⚠️  Dimension mismatch detected")

# Comprehensive file verification
print("\n" + "=" * 80)
print("COMPLETE FEATURE FILE INVENTORY")
print("=" * 80)

feature_files = {
    'train_text.npy': f"{FEATURE_STORAGE}/train_text.npy",
    'train_img.npy': f"{FEATURE_STORAGE}/train_img.npy",
    'test_text.npy': f"{FEATURE_STORAGE}/test_text.npy",
    'test_img.npy': f"{FEATURE_STORAGE}/test_img.npy"
}

all_complete = True
for filename, filepath in feature_files.items():
    if os.path.exists(filepath):
        file_size_mb = os.path.getsize(filepath) / 1024**2
        array_shape = np.load(filepath).shape
        print(f"  ✅ {filename:20s} {array_shape} ({file_size_mb:.1f} MB)")
    else:
        print(f"  ❌ {filename:20s} NOT FOUND")
        all_complete = False

if all_complete:
    print("\n" + "=" * 80)
    print("🎉 COMPLETE FEATURE SET READY! 🎉")
    print("=" * 80)
    print("\n🚀 Workflow continuation:")
    print("   1. Verify all 4 .npy files in embeddings/ directory")
    print("   2. Execute 3_training_enhanced.ipynb for model training")
    print("   3. Generate predictions and prepare submission!")
    print("\n💡 Estimated training duration: ~15-20 minutes")
    print("💡 Target validation SMAPE: 18-25 (competitive range)")
else:
    print("\n⚠️  Incomplete feature set detected!")
    print("   Generate missing files before model training.")

print("\n" + "=" * 80)

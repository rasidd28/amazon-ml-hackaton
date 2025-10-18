# -*- coding: utf-8 -*-
"""
Resumable Heavily Optimized K-Fold Cross-Validation
Features: Auto-save checkpoints, resume from interruption, error recovery
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import gmean
import warnings
import pickle
import os
from pathlib import Path
import traceback
import torch.multiprocessing as mp
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = "/home/me22b1026"
DATA_DIR = f"{PROJECT_ROOT}/cache/dataset"
EMBED_DIR = f"{PROJECT_ROOT}/new/new_embeddings"
OUTPUT_FILE = f"{DATA_DIR}/optimized_predictions.csv"
CHECKPOINT_DIR = f"{DATA_DIR}/checkpoints"

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

NUM_SPLITS = 10
RANDOM_SEEDS = [42, 123, 456, 789, 2024]
MAX_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-5
EARLY_STOP_ROUNDS = 20
APPLY_LOG_TRANSFORM = True

USE_LABEL_SMOOTHING = True
USE_GRADIENT_CLIPPING = True
USE_MIXUP = True
MIXUP_ALPHA = 0.3
USE_CUTOUT = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

def get_bin_edges(data, num_bins):
    """Get bin edges for stratified splitting."""
    q_data = np.quantile(data, np.linspace(0, 1, num_bins + 1))
    return q_data

def apply_mixup(inputs, targets, alpha):
    """MixUp data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(inputs.size(0)).to(inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam

def apply_cutout(inputs, size=16):
    """CutOut data augmentation."""
    # This function is designed for 2D data (e.g., images)
    # Applying it to 1D feature vectors will cause a runtime error.
    # It's commented out in the training loop to prevent that.
    h, w = inputs.shape[1], inputs.shape[2]
    mask = torch.ones_like(inputs)
    y1 = np.random.randint(h)
    x1 = np.random.randint(w)
    y2 = np.clip(y1 + size, 0, h)
    x2 = np.clip(x1 + size, 0, w)
    mask[:, y1:y2, x1:x2] = 0
    return inputs * mask

# ============================================================
# MODEL DEFINITION
# ============================================================
class OptimizedRegressionNet(nn.Module):
    def __init__(self, input_dim):
        super(OptimizedRegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.2)
        self.fc_out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(torch.relu(self.bn4(self.fc4(x))))
        x = self.dropout5(torch.relu(self.bn5(self.fc5(x))))
        x = self.fc_out(x)
        return x

# ============================================================
# RESUMABILITY FUNCTIONS
# ============================================================
def save_checkpoint(fold_idx, seed_idx, trained_models, scalers, scores):
    """Save current progress"""
    checkpoint = {
        'fold_idx': fold_idx,
        'seed_idx': seed_idx,
        'trained_models': [m.state_dict() for m in trained_models],
        'scalers': scalers,
        'scores': scores
    }
    path = f"{CHECKPOINT_DIR}/checkpoint_fold{fold_idx}_seed{seed_idx}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  ?? Checkpoint saved")

def load_latest_checkpoint():
    """Load the most recent checkpoint"""
    checkpoints = list(Path(CHECKPOINT_DIR).glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getctime)
    with open(latest, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"? Resuming from: {latest.name}")
    return checkpoint

# ============================================================
# TRAINING LOGIC
# ============================================================
def train_optimized_model(X_train_fold, X_val_fold, y_train_fold, y_val_fold, fold_idx, seed):
    """Train a single model with optimizations."""
    set_seed(seed)

    # Data scaling
    scaler = RobustScaler()
    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler.transform(X_val_fold)

    # Log transform prices if enabled
    y_train_fold_transformed = np.log1p(y_train_fold) if APPLY_LOG_TRANSFORM else y_train_fold
    y_val_fold_transformed = np.log1p(y_val_fold) if APPLY_LOG_TRANSFORM else y_val_fold
    
    X_train_tensor = torch.tensor(X_train_fold_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train_fold_transformed.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_fold_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val_fold_transformed.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # FIX: Set num_workers=0 to avoid multiprocessing errors
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = OptimizedRegressionNet(X_train_fold_scaled.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    criterion = nn.L1Loss()
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            if USE_MIXUP:
                batch_x, y_a, y_b, lam = apply_mixup(batch_x, batch_y, MIXUP_ALPHA)
            # if USE_CUTOUT:
            #     batch_x = apply_cutout(batch_x)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            if USE_MIXUP:
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            if USE_GRADIENT_CLIPPING:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x_val, batch_y_val in val_loader:
                outputs = model(batch_x_val)
                val_loss += criterion(outputs, batch_y_val).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= EARLY_STOP_ROUNDS:
            break

    # Load best model state and evaluate
    model.load_state_dict(best_model_state)
    model.eval()
    
    val_predictions = []
    with torch.no_grad():
        for batch_x_val, _ in val_loader:
            outputs = model(batch_x_val)
            val_predictions.extend(outputs.cpu().numpy().flatten())
            
    val_predictions = np.array(val_predictions)
    
    if APPLY_LOG_TRANSFORM:
        val_predictions = np.expm1(val_predictions)
    
    score = smape(y_val_fold, val_predictions)
    
    return model, scaler, score

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':
    # FIX: Use 'spawn' start method for multiprocessing compatibility
    mp.set_start_method('spawn', force=True)
    try:
        print("?? Starting Resumable K-Fold Cross-Validation...")
        
        # Load data
        print("? Loading data...")
        df_train = pd.read_csv(f"{DATA_DIR}/train.csv")
        df_test = pd.read_csv(f"{DATA_DIR}/test.csv")
        
        # Load embeddings
        train_text = np.load(f"{EMBED_DIR}/train_text.npy")
        train_img = np.load(f"{EMBED_DIR}/train_img.npy")
        test_text = np.load(f"{EMBED_DIR}/test_text.npy")
        test_img = np.load(f"{EMBED_DIR}/test_img.npy")

        # Combine modalities
        features_train = np.concatenate([train_text, train_img], axis=1)
        features_test = np.concatenate([test_text, test_img], axis=1)

        prices = df_train['price']
        
        # Create bins for stratified splitting
        price_bins = pd.cut(prices, bins=get_bin_edges(prices, NUM_SPLITS), labels=False, include_lowest=True)
        
        # Resumability Logic
        checkpoint = load_latest_checkpoint()
        if checkpoint:
            trained_models = [OptimizedRegressionNet(features_train.shape[1]).to(DEVICE)
                               for _ in checkpoint['trained_models']]
            for model, state in zip(trained_models, checkpoint['trained_models']):
                model.load_state_dict(state)
            feature_scalers = checkpoint['scalers']
            validation_scores = checkpoint['scores']
            start_fold = checkpoint['fold_idx']
            start_seed = checkpoint['seed_idx'] + 1
            print(f"Resuming from Fold {start_fold + 1}, Seed index {start_seed}")
        else:
            trained_models = []
            feature_scalers = []
            validation_scores = []
            start_fold = 0
            start_seed = 0
        
        model_counter = len(trained_models)
        total_models = NUM_SPLITS * len(RANDOM_SEEDS)
        
        skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_SEEDS[0])
        
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features_train, price_bins)):
                if fold_idx < start_fold:
                    continue
                
                print(f"\n{'='*70}")
                print(f"?? FOLD {fold_idx + 1}/{NUM_SPLITS}")
                print(f"{'='*70}")
                
                X_train_fold = features_train[train_idx]
                X_val_fold = features_train[val_idx]
                y_train_fold = prices.iloc[train_idx]
                y_val_fold = prices.iloc[val_idx]
                
                print(f"Train: {len(X_train_fold):,} | Val: {len(X_val_fold):,}")
                
                fold_scores = []
                
                for seed_idx, seed in enumerate(RANDOM_SEEDS):
                    # Skip if resuming and not reached yet
                    if fold_idx == start_fold and seed_idx < start_seed:
                        continue
                    
                    model_counter += 1
                    print(f"  Model {model_counter}/{total_models} (Seed {seed})", end=" ? ")
                    
                    try:
                        model, scaler, score = train_optimized_model(
                            X_train_fold, X_val_fold, y_train_fold, y_val_fold,
                            fold_idx, seed
                        )
                        
                        trained_models.append(model)
                        feature_scalers.append(scaler)
                        validation_scores.append(score)
                        fold_scores.append(score)
                        
                        print(f"SMAPE: {score:.2f}")
                        
                        # Save checkpoint after each model
                        save_checkpoint(fold_idx, seed_idx, trained_models,
                                       feature_scalers, validation_scores)
                    
                    except Exception as e:
                        print(f"\n? Error training model: {e}")
                        traceback.print_exc()
                        print("?? Progress saved. Rerun script to resume.")
                        raise
                
                fold_avg = np.mean(fold_scores) if fold_scores else 0
                fold_std = np.std(fold_scores) if fold_scores else 0
                print(f"  ?? Fold Average: {fold_avg:.2f} +/- {fold_std:.2f}")
        
        except KeyboardInterrupt:
            print("\n??  Training interrupted by user")
            print("?? Progress saved. Rerun script to resume from checkpoint.")
            exit(0)
        except Exception as e:
            print(f"\n? Fatal error: {e}")
            traceback.print_exc()
            print("?? Progress saved. Fix error and rerun to resume.")
            exit(1)
            
        print("\n? All models trained!")
        
        # Make predictions on test set
        print("?? Generating final predictions...")
        final_predictions = []
        for model, scaler in zip(trained_models, feature_scalers):
            test_scaled = scaler.transform(features_test)
            test_tensor = torch.tensor(test_scaled, dtype=torch.float32).to(DEVICE)
            
            model.eval()
            with torch.no_grad():
                preds = model(test_tensor).cpu().numpy().flatten()
            
            if APPLY_LOG_TRANSFORM:
                preds = np.expm1(preds)
            final_predictions.append(preds)
        
        # Aggregate predictions using geometric mean
        final_preds_gmean = gmean(np.clip(final_predictions, 1e-6, None), axis=0)
        
        # Save predictions
        submission_df = pd.DataFrame({'sample_id': df_test['sample_id'], 'price': final_preds_gmean})
        submission_df.to_csv(OUTPUT_FILE, index=False)
        print(f"?? Predictions saved to {OUTPUT_FILE}")
        
        # Clean up checkpoints after successful completion
        print("\n?? Cleaning up checkpoints...")
        for ckpt in Path(CHECKPOINT_DIR).glob("checkpoint_*.pkl"):
            ckpt.unlink()
        print("? All done!")

    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please ensure the data and embedding files exist at the specified paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
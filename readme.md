# 🏆 Amazon ML Hackathon - Multimodal Price Prediction System

> **Final Achievement**: Secured **Rank 144 out of 7,107 teams** with a SMAPE score of **44.370**

An end-to-end multimodal machine learning pipeline that predicts product prices using both textual descriptions and visual imagery from Amazon product catalogs. This project combines state-of-the-art NLP and computer vision with advanced deep learning techniques including K-Fold Cross-Validation, ensemble learning, and checkpoint-based recovery.

---

## 👥 Team Members

- **Puneeth P**
- **Venu Madhav Reddy**
- **Bhargav Sai Teja**
- **M Rahul Siddhartha** 

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technical Details](#-technical-details)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)

---

## 🎯 Problem Statement

**Objective**: Predict product prices based on:
- 📝 Product catalog text descriptions (item names, specifications, features)
- 🖼️ Product images (visual appearance, packaging, style)

**Challenges**:
- Handle missing data gracefully
- Extract meaningful features from unstructured multimodal data
- Achieve competitive prediction accuracy
- Ensure training resilience with checkpoint recovery

**Evaluation Metric**: **SMAPE** (Symmetric Mean Absolute Percentage Error)

---

## 🏗️ System Architecture

### Phase 1: Exploratory Data Analysis (EDA)
Comprehensive statistical analysis including:
- Price distribution analysis with outlier detection
- Text content analysis (character/token counts, structural features)
- Image availability validation
- Feature correlation studies

### Phase 2: Multimodal Feature Extraction

#### Text Features (1024-D)
- **Model**: BAAI/bge-large-en-v1.5
- **Output**: Dense semantic embeddings
- **Batch Processing**: 128 samples per batch
- **Normalization**: L2 normalization

#### Vision Features (768-D)
- **Model**: CLIP-ViT-Large-Patch14
- **Output**: Visual embeddings
- **Parallel Downloads**: 64 concurrent workers
- **Batch Processing**: 32 images per batch
- **Checkpoint Recovery**: Every 5,000 samples

### Phase 3: Advanced Model Training

**Deep Neural Network Architecture**:
```
Input:    1792-D (1024 text + 768 vision)
         ↓
Layer 1:  2048 neurons → BatchNorm → Dropout(0.2) → ReLU
Layer 2:  1024 neurons → BatchNorm → Dropout(0.2) → ReLU
Layer 3:  512 neurons  → BatchNorm → Dropout(0.2) → ReLU
Layer 4:  256 neurons  → BatchNorm → Dropout(0.2) → ReLU
Layer 5:  128 neurons  → BatchNorm → Dropout(0.2) → ReLU
         ↓
Output:   1 neuron (price prediction)

Parameters: ~5.5M trainable
```

**Training Strategy**:
- **K-Fold CV**: 10 folds with stratified splitting
- **Multiple Seeds**: 5 different seeds [42, 123, 456, 789, 2024]
- **Total Models**: 50 models (10 folds × 5 seeds)
- **Ensemble**: Geometric mean aggregation

---

## ✨ Key Features

### 🔄 Checkpoint-Based Resumability
- Automatic checkpoint saving after each model
- Resume from exact interruption point
- No data loss on crashes or manual stops
- Progress tracking with persistent storage

### 🎯 Advanced Training Techniques
- **MixUp Augmentation** (α=0.3): Improves robustness
- **AdamW Optimizer**: Decoupled weight decay
- **StepLR Scheduler**: Learning rate decay
- **Early Stopping**: 20 epochs patience
- **Gradient Clipping**: Max norm = 1.0

### 📊 Robust Cross-Validation
- Stratified K-Fold ensures balanced price distribution
- Multiple random seeds reduce initialization bias
- Comprehensive validation scoring

### 🚀 Performance Optimizations
- Parallel image downloading (64 threads)
- GPU-accelerated batch processing
- Memory management with cache clearing
- Feature embedding caching

---

## 📁 Project Structure

```
amazon-ml-hackathon/
├── dataset/
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Testing data
│   └── optimized_predictions.csv      # Final submission
│
├── images/                            # Downloaded product images
│
├── embeddings/                        # Generated features
│   ├── train_text.npy                 # (N, 1024)
│   ├── train_img.npy                  # (N, 768)
│   ├── test_text.npy                  # (M, 1024)
│   └── test_img.npy                   # (M, 768)
│
├── scripts/
│   ├── eda.py                         # Exploratory analysis
│   ├── embeddings.py                  # Feature extraction
│   └── submission.py                  # Training & prediction
│
├── visualizations/
│   ├── price_distribution.png
│   ├── text_analysis.png
│   └── correlations.png
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/amazon-ml-hackathon.git
cd amazon-ml-hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt
```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
pillow>=10.0.0
requests>=2.31.0
tqdm>=4.65.0
```

---

## 🚀 Usage

### Step 1: Exploratory Data Analysis

```bash
python scripts/eda.py
```

**Outputs**:
- Price distribution visualizations
- Text content analysis plots
- Feature correlation heatmaps
- Statistical summaries

### Step 2: Generate Multimodal Features

```bash
python scripts/embeddings.py
```

**Process**:
- Downloads product images (with retry mechanism)
- Generates text embeddings using BGE-large
- Generates image embeddings using CLIP
- Saves features as `.npy` files

**Time**: ~30 minutes (depends on dataset size and network)

### Step 3: Train Models & Generate Predictions

```bash
python scripts/submission.py
```

**Process**:
- Trains 50 models (10 folds × 5 seeds)
- Checkpoint-based resumability (can interrupt anytime)
- Ensemble predictions using geometric mean
- Generates `optimized_predictions.csv`

**Time**: ~15 hours (resumable at any point)

**Note**: If interrupted, simply rerun the command - it will automatically resume from the last checkpoint!

---

## 🔬 Technical Details

### Text Encoding
- **Model**: BAAI/bge-large-en-v1.5
- **Dimension**: 1024-D dense vectors
- **Advantage**: SOTA performance on MTEB leaderboard
- **Captures**: Semantic meaning, brand signals, specifications

### Vision Encoding
- **Model**: CLIP-ViT-Large-Patch14 (OpenAI)
- **Dimension**: 768-D dense vectors
- **Advantage**: Vision-language alignment, zero-shot capability
- **Captures**: Visual aesthetics, packaging, design elements

### Model Training
- **Optimizer**: AdamW (lr=5e-4, weight_decay=5e-5)
- **Loss**: L1Loss (MAE) - better for price prediction
- **Regularization**: Dropout(0.2) + BatchNorm + Weight Decay
- **Augmentation**: MixUp (α=0.3)
- **Target Transform**: Log1p (handles skewed distribution)

### Ensemble Strategy
- **Method**: Geometric mean (better for multiplicative processes)
- **Models**: 50 diverse models
- **Benefit**: Reduced variance, improved generalization

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Final Rank** | 144 / 7,107 teams |
| **SMAPE Score** | 44.370 |
| **Top 2%** | ✅ |
| **Models Trained** | 50 |
| **Training Time** | ~15 hours |

### Performance Analysis
- Successfully handled missing images with zero vectors
- Robust to price distribution skewness
- Strong generalization across different product categories
- Effective multimodal fusion strategy

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Attention-based fusion mechanisms
- [ ] Transformer-based cross-modal attention
- [ ] Advanced ensemble methods (stacking, boosting)

### Feature Engineering
- [ ] Extract numeric attributes from text (prices, dimensions)
- [ ] Brand and category one-hot encoding
- [ ] Image quality scores and color histograms
- [ ] Historical price data integration

### Optimization
- [ ] Hyperparameter tuning with Optuna
- [ ] Neural Architecture Search (NAS)
- [ ] Model compression (quantization, pruning)
- [ ] Mixed precision training (FP16)


---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Pre-trained Models
- **BAAI** - BGE text embedding model
- **OpenAI** - CLIP vision-language model

### Frameworks & Libraries
- **PyTorch** - Deep learning framework
- **Hugging Face** - Transformers library
- **Scikit-learn** - ML utilities

### Competition
- **Amazon ML Challenge** - For organizing this exciting hackathon
- **Our Team** - For the collaborative effort and dedication



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("AMAZON ML CHALLENGE - DATA EXPLORATION & ANALYSIS")
print("=" * 80)

# ====================================================================================
# SECTION 1: DATA LOADING
# ====================================================================================
print("\n[STEP 1] Loading datasets from disk...")

DATA_PATH = "/home/ME22B1026/amazon_ml_hackathon"
training_data = pd.read_csv(f"{DATA_PATH}/dataset/train.csv")
testing_data = pd.read_csv(f"{DATA_PATH}/dataset/test.csv")

print(f"Training set size: {len(training_data):,} records")
print(f"Testing set size: {len(testing_data):,} records")

# ====================================================================================
# SECTION 2: DATASET STRUCTURE EXAMINATION
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 2] EXAMINING DATASET STRUCTURE")
print("=" * 80)

print("\nTraining Data Information:")
training_data.info()

print("\nInitial rows from training set:")
print(training_data.head())

print("\nAvailable columns:")
for idx, column in enumerate(training_data.columns, 1):
    print(f"  {idx}. {column}")

# ====================================================================================
# SECTION 3: NULL VALUE DETECTION
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 3] NULL VALUE ANALYSIS")
print("=" * 80)

train_nulls = training_data.isnull().sum()
test_nulls = testing_data.isnull().sum()

print("\nNull values in training data:")
for column in training_data.columns:
    null_count = train_nulls[column]
    null_percentage = (null_count / len(training_data)) * 100
    status = "✓ Complete" if null_count == 0 else f"{null_count:,} nulls ({null_percentage:.2f}%)"
    print(f"  {column}: {status}")

print("\nNull values in testing data:")
for column in testing_data.columns:
    null_count = test_nulls[column]
    null_percentage = (null_count / len(testing_data)) * 100
    status = "✓ Complete" if null_count == 0 else f"{null_count:,} nulls ({null_percentage:.2f}%)"
    print(f"  {column}: {status}")

# ====================================================================================
# SECTION 4: PRICE DISTRIBUTION STUDY
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 4] PRICE VARIABLE EXPLORATION")
print("=" * 80)

print("\nStatistical summary of prices:")
print(training_data['price'].describe())

price_min = training_data['price'].min()
price_max = training_data['price'].max()
price_avg = training_data['price'].mean()
price_mid = training_data['price'].median()
price_std = training_data['price'].std()

print(f"\nPrice metrics:")
print(f"  Minimum value: ${price_min:.2f}")
print(f"  Maximum value: ${price_max:.2f}")
print(f"  Average: ${price_avg:.2f}")
print(f"  Median: ${price_mid:.2f}")
print(f"  Standard deviation: ${price_std:.2f}")

# Outlier detection using IQR
first_quartile = training_data['price'].quantile(0.25)
third_quartile = training_data['price'].quantile(0.75)
interquartile_range = third_quartile - first_quartile
lower_bound = first_quartile - 1.5 * interquartile_range
upper_bound = third_quartile + 1.5 * interquartile_range
outlier_mask = (training_data['price'] < lower_bound) | (training_data['price'] > upper_bound)
outlier_count = outlier_mask.sum()
outlier_pct = (outlier_count / len(training_data)) * 100

print(f"\nOutliers detected: {outlier_count} ({outlier_pct:.2f}% of data)")

# Create price visualization
fig = plt.figure(figsize=(16, 12))

# Standard histogram
ax1 = plt.subplot(2, 2, 1)
ax1.hist(training_data['price'], bins=80, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(price_avg, color='red', linestyle='dashed', linewidth=2, label=f'Mean: ${price_avg:.2f}')
ax1.axvline(price_mid, color='green', linestyle='dashed', linewidth=2, label=f'Median: ${price_mid:.2f}')
ax1.set_xlabel('Price (USD)')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Product Prices')
ax1.legend()
ax1.grid(alpha=0.3)

# Logarithmic scale histogram
ax2 = plt.subplot(2, 2, 2)
log_prices = np.log1p(training_data['price'])
ax2.hist(log_prices, bins=80, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Log(Price + 1)')
ax2.set_ylabel('Count')
ax2.set_title('Price Distribution - Logarithmic Scale')
ax2.grid(alpha=0.3)

# Box and whisker plot
ax3 = plt.subplot(2, 2, 3)
box_data = ax3.boxplot(training_data['price'], vert=True, patch_artist=True)
box_data['boxes'][0].set_facecolor('lightblue')
ax3.set_ylabel('Price (USD)')
ax3.set_title('Price Box Plot - Outlier Visualization')
ax3.grid(axis='y', alpha=0.3)

# Cumulative distribution function
ax4 = plt.subplot(2, 2, 4)
price_sorted = np.sort(training_data['price'])
cumulative_pct = np.arange(1, len(price_sorted) + 1) / len(price_sorted) * 100
ax4.plot(price_sorted, cumulative_pct, color='darkviolet', linewidth=2)
ax4.set_xlabel('Price (USD)')
ax4.set_ylabel('Cumulative Percentage')
ax4.set_title('Cumulative Price Distribution')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DATA_PATH}/price_distribution_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✓ Visualization saved: price_distribution_analysis.png")

# ====================================================================================
# SECTION 5: TEXT CONTENT EXPLORATION
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 5] CATALOG TEXT CONTENT ANALYSIS")
print("=" * 80)

# Calculate text metrics
training_data['char_count'] = training_data['catalog_content'].fillna("").astype(str).str.len()
training_data['token_count'] = training_data['catalog_content'].fillna("").astype(str).str.split().str.len()

print("\nCharacter count statistics:")
print(training_data['char_count'].describe())

print("\nToken count statistics:")
print(training_data['token_count'].describe())

# Display sample entries
print("\nSample catalog entries:")
for idx in range(3):
    print(f"\n[Sample {idx + 1}]")
    sample_text = training_data['catalog_content'].iloc[idx]
    print(sample_text[:300] + "..." if len(str(sample_text)) > 300 else sample_text)

# Extract structural features
def detect_content_features(content):
    """Identify structural elements in catalog text"""
    features = []
    text_lower = str(content).lower()
    
    if 'item name:' in text_lower:
        features.append('contains_item_name')
    
    bullet_matches = text_lower.count('bullet point')
    if bullet_matches > 0:
        features.append(f'bullet_count_{bullet_matches}')
    
    if 'value:' in text_lower:
        features.append('contains_value')
    
    if 'unit:' in text_lower:
        features.append('contains_unit')
    
    return features

training_data['content_features'] = training_data['catalog_content'].apply(detect_content_features)

# Aggregate feature statistics
feature_stats = {
    'contains_item_name': sum('contains_item_name' in f for f in training_data['content_features']),
    'contains_bullet_points': sum(any('bullet_count_' in x for x in f) for f in training_data['content_features']),
    'contains_value': sum('contains_value' in f for f in training_data['content_features']),
    'contains_unit': sum('contains_unit' in f for f in training_data['content_features']),
}

print("\nStructural feature detection:")
for feature_name, feature_count in feature_stats.items():
    feature_pct = (feature_count / len(training_data)) * 100
    print(f"  {feature_name}: {feature_count:,} records ({feature_pct:.1f}%)")

# Visualize text metrics
fig = plt.figure(figsize=(18, 6))

# Character count distribution
ax1 = plt.subplot(1, 3, 1)
ax1.hist(training_data['char_count'], bins=60, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Character Count')
ax1.set_ylabel('Frequency')
ax1.set_title('Text Length Distribution (Characters)')
ax1.grid(alpha=0.3)

# Token count distribution
ax2 = plt.subplot(1, 3, 2)
ax2.hist(training_data['token_count'], bins=60, color='darkorange', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Token Count')
ax2.set_ylabel('Frequency')
ax2.set_title('Text Length Distribution (Tokens)')
ax2.grid(alpha=0.3)

# Feature presence comparison
ax3 = plt.subplot(1, 3, 3)
feature_names = list(feature_stats.keys())
feature_values = list(feature_stats.values())
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
bars = ax3.bar(feature_names, feature_values, color=colors, edgecolor='black', alpha=0.8)
ax3.set_ylabel('Record Count')
ax3.set_title('Content Feature Prevalence')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{DATA_PATH}/text_content_analysis.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✓ Visualization saved: text_content_analysis.png")

# ====================================================================================
# SECTION 6: IMAGE DATA AVAILABILITY
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 6] IMAGE LINK AVAILABILITY CHECK")
print("=" * 80)

# Check image presence
training_data['image_available'] = training_data['image_link'].notna()
testing_data['image_available'] = testing_data['image_link'].notna()

train_image_count = training_data['image_available'].sum()
test_image_count = testing_data['image_available'].sum()
train_image_pct = (train_image_count / len(training_data)) * 100
test_image_pct = (test_image_count / len(testing_data)) * 100

print(f"\nImage availability in training set: {train_image_count:,} ({train_image_pct:.1f}%)")
print(f"Image availability in testing set: {test_image_count:,} ({test_image_pct:.1f}%)")

# Parse URL characteristics
def parse_image_url(url_string):
    """Extract metadata from image URL"""
    if pd.isna(url_string):
        return None
    
    url_str = str(url_string)
    metadata = {
        'source': 'amazon' if 'amazon' in url_str.lower() else 'external',
        'has_query_params': '?' in url_str,
    }
    return metadata

training_data['url_metadata'] = training_data['image_link'].apply(parse_image_url)

# Summarize URL sources
source_distribution = training_data['url_metadata'].apply(
    lambda x: x['source'] if x else 'no_image'
).value_counts()

print("\nImage source distribution:")
for source_type, count in source_distribution.items():
    pct = (count / len(training_data)) * 100
    print(f"  {source_type}: {count:,} ({pct:.1f}%)")

# ====================================================================================
# SECTION 7: FEATURE CORRELATION ANALYSIS
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 7] RELATIONSHIP ANALYSIS")
print("=" * 80)

# Create scatter plots
fig = plt.figure(figsize=(16, 6))

# Price vs character count
ax1 = plt.subplot(1, 2, 1)
ax1.scatter(training_data['char_count'], training_data['price'], 
           alpha=0.4, s=15, color='navy')
ax1.set_xlabel('Character Count')
ax1.set_ylabel('Price (USD)')
ax1.set_title('Price vs Character Count')
ax1.set_xlim(0, training_data['char_count'].quantile(0.98))
ax1.grid(alpha=0.3)

# Price vs token count
ax2 = plt.subplot(1, 2, 2)
ax2.scatter(training_data['token_count'], training_data['price'], 
           alpha=0.4, s=15, color='darkred')
ax2.set_xlabel('Token Count')
ax2.set_ylabel('Price (USD)')
ax2.set_title('Price vs Token Count')
ax2.set_xlim(0, training_data['token_count'].quantile(0.98))
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DATA_PATH}/feature_correlations.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"\n✓ Visualization saved: feature_correlations.png")

# Compute correlation coefficients
char_correlation = training_data[['char_count', 'price']].corr().iloc[0, 1]
token_correlation = training_data[['token_count', 'price']].corr().iloc[0, 1]

print(f"\nPearson correlation with price:")
print(f"  Character count: {char_correlation:.4f}")
print(f"  Token count: {token_correlation:.4f}")

# ====================================================================================
# SECTION 8: FINAL SUMMARY
# ====================================================================================
print("\n" + "=" * 80)
print("[STEP 8] ANALYSIS SUMMARY & MODELING STRATEGY")
print("=" * 80)

print(f"""
Dataset Characteristics:
  • Training records: {len(training_data):,}
  • Testing records: {len(testing_data):,}
  • Target range: ${price_min:.2f} to ${price_max:.2f}

Principal Observations:
  ✓ {train_image_count:,} images present in training data
  ✓ Character count range: {int(training_data['char_count'].min())} to {int(training_data['char_count'].max())}
  ✓ Price distribution shows right skew (median < mean)
  ✓ Detected {outlier_count} potential outliers using IQR method

Proposed Modeling Strategy:
  1. Implement multimodal architecture (text + image)
  2. Text encoding: BAAI/bge-large-en-v1.5 (1024 dimensions)
  3. Image encoding: CLIP-ViT-Large (768 dimensions)
  4. Handle missing images with zero-vector fallback
  5. Apply StandardScaler for feature scaling
  6. Evaluation metric: SMAPE (competition requirement)
  7. Consider log transformation for price normalization

Workflow Progression:
  → Execute 2_embeddings.ipynb for feature extraction
  → Execute 3_training.ipynb for model development
""")

print("\n✅ EXPLORATORY DATA ANALYSIS COMPLETE!")
print("=" * 80)

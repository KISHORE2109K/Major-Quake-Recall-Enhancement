# =======================================================
# 1. SETUP & IMPORTS
# =======================================================
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("--- Project Start: Synthetic Data for Earthquake Prediction ---")

# =======================================================
# 2. DATA PREPARATION
# =======================================================
print("\n[Phase 1] Loading and Preparing Real Earthquake Data...")

# Load the dataset
try:
    real_data = pd.read_csv('data/query.csv')
except FileNotFoundError:
    print("Error: 'data/query.csv' not found. Please make sure the dataset is in the correct folder.")
    exit()

# --- Feature Engineering & Cleaning ---
# Convert 'time' column to a proper datetime format
real_data['time'] = pd.to_datetime(real_data['time'])

# We'll use a curated list of features relevant for prediction
features_to_use = [
    'latitude', 'longitude', 'depth', 'mag', 'magType', 'nst', 'gap', 'rms'
]
real_data = real_data[features_to_use].copy()
real_data.dropna(inplace=True)

# --- NEW STEP: One-Hot Encode Categorical Features ---
# RandomForestClassifier cannot handle text. We convert 'magType' into numerical columns.
real_data = pd.get_dummies(real_data, columns=['magType'], prefix='magType')


# --- Define Our Prediction Target ---
# Our goal is to predict major earthquakes. Let's define "major" as magnitude > 5.5
# We create a new target column: 1 for a major quake, 0 for a minor one.
real_data['is_major_quake'] = (real_data['mag'] > 5.5).astype(int)
real_data.drop('mag', axis=1, inplace=True) # Drop original 'mag' as it's our target

print("Data Prepared. Shape of real data:", real_data.shape)
print("Distribution of real data (0 = Minor, 1 = Major):")
print(real_data['is_major_quake'].value_counts(normalize=True))

# Split data for a final, untouched test set
X = real_data.drop('is_major_quake', axis=1)
y = real_data['is_major_quake']
# 'stratify=y' is important for imbalanced data to keep proportions the same in train/test sets
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Combine the training parts back for the synthesizer to learn from
train_data_real = pd.concat([X_train_real, y_train_real], axis=1)
# =======================================================
# 3. SYNTHETIC DATA GENERATION
# =======================================================
print("\n[Phase 2] Training CTGAN to Generate Synthetic Data (This may take 10-20 minutes)...")

# Create the data "blueprint" (metadata)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train_data_real)

# Initialize the CTGAN model
synthesizer = CTGANSynthesizer(metadata, epochs=300, verbose=True)

# Train the synthesizer on our real training data
synthesizer.fit(train_data_real)

# --- Generate a large batch of synthetic data ---
# The model will generate data based on the original imbalanced distribution
print("Generating a large batch of synthetic data...")
large_synthetic_sample = synthesizer.sample(num_rows=50000) # Generate 50,000 new rows

# --- Filter to get only the RARE class ---
# We will filter this large batch to find the synthetic major quakes
synthetic_major_quakes_filtered = large_synthetic_sample[
    large_synthetic_sample['is_major_quake'] == 1
]
print(f"Found {len(synthetic_major_quakes_filtered)} synthetic major quakes in the batch.")


# --- Create the Augmented Dataset ---
# Determine how many synthetic samples we need to balance the dataset
num_minor_quakes = train_data_real['is_major_quake'].value_counts()[0]
num_major_quakes_real = train_data_real['is_major_quake'].value_counts()[1]
num_to_add = num_minor_quakes - num_major_quakes_real

# Take only the number of synthetic samples we need
synthetic_major_quakes = synthetic_major_quakes_filtered.head(num_to_add)

# This dataset contains all original training data PLUS the new synthetic major quakes
augmented_data = pd.concat([train_data_real, synthetic_major_quakes])

print("Augmented dataset created. New distribution:")
print(augmented_data['is_major_quake'].value_counts(normalize=True))

# =======================================================
# 4. EVALUATION: BASELINE vs. ENHANCED MODEL
# =======================================================
print("\n[Phase 3] Evaluating Models...")

# --- Model 1: Trained on Original Imbalanced Data ---
print("\n--- Training Baseline Model on Original Data ---")
X_train_baseline = train_data_real.drop('is_major_quake', axis=1)
y_train_baseline = train_data_real['is_major_quake']

baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train_baseline, y_train_baseline)

print("--- Baseline Model Performance ---")
baseline_predictions = baseline_model.predict(X_test_real)
print(classification_report(y_test_real, baseline_predictions, target_names=['Minor Quake', 'Major Quake']))

# --- Model 2: Trained on Augmented (Real + Synthetic) Data ---
print("\n--- Training Enhanced Model on Augmented Data ---")
X_train_augmented = augmented_data.drop('is_major_quake', axis=1)
y_train_augmented = augmented_data['is_major_quake']

augmented_model = RandomForestClassifier(random_state=42)
augmented_model.fit(X_train_augmented, y_train_augmented)

print("--- Enhanced Model Performance (with Synthetic Data) ---")
augmented_predictions = augmented_model.predict(X_test_real)
print(classification_report(y_test_real, augmented_predictions, target_names=['Minor Quake', 'Major Quake']))

print("\n--- Project Complete ---")
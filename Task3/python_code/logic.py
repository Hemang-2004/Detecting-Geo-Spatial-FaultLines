import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from PIL import Image
import imageio

def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def k_fold_feedback_loop(X, y, k=5, save_dir_images="/home/vasu/Documents/A3/images", save_dir_gif="/home/vasu/Documents/A3/gif"):
    # Ensure directories exist
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_gif, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    gif_images = []
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        # Split data
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create model
        model = create_model(X_train.shape[1], y.shape[1])

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=4,
            batch_size=32,
            verbose=0
        )

        # Evaluate model
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy:.4f}")

        # Plot Accuracy and Loss
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy Plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        axes[0].set_title("Model Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True)

        # Loss Plot
        axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
        axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o')
        axes[1].set_title("Model Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)

        # Save the plot
        image_path = os.path.join(save_dir_images, f"fold_{fold}_performance.png")
        plt.savefig(image_path)
        plt.close()

        # Add to GIF sequence
        img = Image.open(image_path)
        gif_images.append(img)

    # Create GIF
    gif_path = os.path.join(save_dir_gif, "k_fold_performance.gif")
    imageio.mimsave(gif_path, gif_images, duration=1)

    # Print summary
    print("\nK-Fold Cross-Validation Summary:")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")

# Load and preprocess data
relevant_columns = [
    'district_id', 'damage_grade', 'technical_solution_proposed', 'has_repair_started',
    'has_damage_foundation', 'has_damage_roof', 'has_damage_corner_separation',
    'has_damage_diagonal_cracking', 'has_damage_in_plane_failure', 'has_damage_out_of_plane_failure',
    'has_damage_out_of_plane_walls_ncfr_failure', 'has_damage_gable_failure',
    'has_damage_delamination_failure', 'has_damage_column_failure', 'has_damage_beam_failure',
    'has_damage_infill_partition_failure', 'has_damage_staircase', 'has_damage_parapet',
    'has_damage_cladding_glazing', 'has_geotechnical_risk', 'has_geotechnical_risk_land_settlement',
    'has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_liquefaction',
    'has_geotechnical_risk_landslide', 'has_geotechnical_risk_rock_fall',
    'has_geotechnical_risk_flood', 'has_geotechnical_risk_other'
]

# Load data
data = pd.read_csv("/home/vasu/Documents/A3/csv_files/csv_building_damage_assessment.csv", low_memory=False)
data = data[relevant_columns]

# Drop rows with missing 'damage_grade'
data = data.dropna(subset=['damage_grade'])

# Encode the target variable 'damage_grade'
label_encoder = LabelEncoder()
data['damage_grade'] = label_encoder.fit_transform(data['damage_grade'])

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
data_imputed = imputer.fit_transform(data)
data = pd.DataFrame(data_imputed, columns=data.columns)

# Encode non-numeric columns
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = LabelEncoder().fit_transform(data[column])

# Separate features and target
X = data.drop(['damage_grade'], axis=1)
y = data['damage_grade']

# One-hot encode the target variable
y = to_categorical(y)

# Run k-fold feedback loop
k_fold_feedback_loop(X, y, k=10)

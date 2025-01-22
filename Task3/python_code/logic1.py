import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Create directories for saving images
base_dir = "images"
os.makedirs(base_dir, exist_ok=True)

# Load relevant columns from dataset
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

# Impute missing values for features
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

# Run visualizations for 4 runs with different plot types
for run in range(1, 5):
    run_dir = os.path.join(base_dir, f"run_{run}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Distribution of damage grades with different plot types
    plt.figure(figsize=(10, 6))
    if run == 1:
        sns.countplot(x='damage_grade', data=data, palette='Set2')
        plt.title('Distribution of Damage Grades (Bar Plot)')
    elif run == 2:
        data['damage_grade'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
        plt.title('Distribution of Damage Grades (Pie Chart)')
    elif run == 3:
        sns.countplot(y='damage_grade', data=data, palette='Set2')
        plt.title('Distribution of Damage Grades (Horizontal Bar Plot)')
    elif run == 4:
        data['damage_grade'].value_counts().plot(kind='area', figsize=(10, 6))
        plt.title('Distribution of Damage Grades (Area Plot)')
    plt.xlabel('Damage Grade')
    plt.ylabel('Count')
    plt.savefig(os.path.join(run_dir, 'damage_grade_distribution.png'))
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Features')
    plt.savefig(os.path.join(run_dir, 'correlation_heatmap.png'))
    plt.close()

    # 3. Most frequent damage types with different plot types
    damage_columns = [
        'has_damage_foundation', 'has_damage_roof', 'has_damage_corner_separation',
        'has_damage_diagonal_cracking', 'has_damage_in_plane_failure', 'has_damage_out_of_plane_failure',
        'has_damage_out_of_plane_walls_ncfr_failure', 'has_damage_gable_failure',
        'has_damage_delamination_failure', 'has_damage_column_failure', 'has_damage_beam_failure',
        'has_damage_infill_partition_failure', 'has_damage_staircase', 'has_damage_parapet',
        'has_damage_cladding_glazing'
    ]
    damage_data = data[damage_columns].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    if run == 1:
        sns.barplot(x=damage_data.index, y=damage_data.values, palette='Blues_d')
        plt.title('Most Frequent Damage Types (Bar Plot)')
    elif run == 2:
        damage_data.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
        plt.title('Most Frequent Damage Types (Pie Chart)')
    elif run == 3:
        sns.barplot(y=damage_data.index, x=damage_data.values, palette='Blues_d')
        plt.title('Most Frequent Damage Types (Horizontal Bar Plot)')
    elif run == 4:
        damage_data.plot(kind='line', figsize=(10, 6))
        plt.title('Most Frequent Damage Types (Line Plot)')
    plt.xlabel('Damage Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(run_dir, 'most_frequent_damage_types.png'))
    plt.close()

    # 4. PCA visualization with different markers
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['damage_grade'] = y

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='damage_grade', data=pca_df, palette='Set1', marker="o" if run == 1 else ("s" if run == 2 else "D"))
    plt.title(f'PCA of Building Damage Assessment Data (Run {run})')
    plt.savefig(os.path.join(run_dir, 'pca_visualization.png'))
    plt.close()

    # 5. Clustering (KMeans) with different colors for each run
    kmeans = KMeans(n_clusters=4, random_state=42)
    X_clustered = kmeans.fit_predict(X)
    pca_df['Cluster'] = X_clustered

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=pca_df, marker="o" if run == 1 else ("s" if run == 2 else "D"))
    plt.title(f'Clustering of Building Damage Assessment Data (Run {run})')
    plt.savefig(os.path.join(run_dir, 'clustering_visualization.png'))
    plt.close()

    # 6. Damage grade by clusters with different plot types
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='damage_grade', data=pca_df, palette='Set3')
    plt.title(f'Damage Grade Distribution Across Clusters (Run {run})')
    plt.xlabel('Cluster')
    plt.ylabel('Damage Grade')
    plt.savefig(os.path.join(run_dir, 'damage_grade_by_cluster.png'))
    plt.close()

    # 7. Most frequent geotechnical risks with different plot types
    geotechnical_risk_columns = [
        'has_geotechnical_risk', 'has_geotechnical_risk_land_settlement',
        'has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_liquefaction',
        'has_geotechnical_risk_landslide', 'has_geotechnical_risk_rock_fall',
        'has_geotechnical_risk_flood', 'has_geotechnical_risk_other'
    ]
    geotechnical_risks = data[geotechnical_risk_columns].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    if run == 1:
        geotechnical_risks.plot(kind='bar', color='green')
        plt.title('Most Frequent Geotechnical Risks (Bar Plot)')
    elif run == 2:
        geotechnical_risks.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
        plt.title('Most Frequent Geotechnical Risks (Pie Chart)')
    elif run == 3:
        sns.barplot(x=geotechnical_risks.index, y=geotechnical_risks.values, palette='viridis')
        plt.title('Most Frequent Geotechnical Risks (Bar Plot)')
    elif run == 4:
        geotechnical_risks.plot(kind='line', figsize=(10, 6))
        plt.title('Most Frequent Geotechnical Risks (Line Plot)')
    plt.xlabel('Geotechnical Risk')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(run_dir, 'geotechnical_risks.png'))
    plt.close()

print("Visualizations generated for all 4 runs!")

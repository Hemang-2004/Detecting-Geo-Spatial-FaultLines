import pandas as pd
import plotly.express as px

# Load the dataset
data = pd.read_csv('/home/vasu/Documents/data_visualization/dataset/csv_building_damage_assessment.csv', low_memory=False)

# Fill missing values to avoid issues
data = data.fillna('Unknown')  # Replace NaN with 'Unknown' for visualization

# Select the columns for the hierarchy
hierarchy_columns = [
    'district_id',            # Outermost layer: Districts
    'ward_id',                # Next layer: Wards
    'damage_grade',           # Next layer: Damage Grade
    'has_geotechnical_risk',  # Next layer: Geotechnical Risk (Yes/No/Unknown)
]

# Ensure columns used for hierarchy are treated as strings
data[hierarchy_columns] = data[hierarchy_columns].astype(str)

# Add a numeric column for size (e.g., count of entries in each hierarchy level)
data['Count'] = 1

# Create the sunburst visualization
fig = px.sunburst(
    data,
    path=hierarchy_columns,  # Define the hierarchy
    values='Count',          # Use the count column to size the sectors
    color='damage_grade',    # Color the sectors based on damage grade
    title="Detailed Sunburst of Earthquake Damage Assessment",
    color_discrete_sequence=px.colors.qualitative.Set3  # Custom color palette
)

# Customize the layout for better appearance
fig.update_layout(
    title_font_size=24,
    title_x=0.5,
    margin=dict(t=50, l=25, r=25, b=25)
)

# Show the graph
fig.show()

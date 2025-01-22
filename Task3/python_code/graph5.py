import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('/home/vasu/Documents/A3/csv_files/csv_building_damage_assessment.csv')

# Preprocess data
df = df.dropna(subset=['damage_grade', 'has_geotechnical_risk'])  # Remove rows with missing essential data

# Create a new column for damage category (based on multiple damage-related columns)
df['Damage Category'] = df.apply(
    lambda row: 'Severe' if row['damage_foundation_severe'] == 'Severe-Extreme-(<1/3)' 
    or row['damage_roof_severe'] == 'Severe-Extreme-(<1/3)' else 'Moderate' if row['damage_foundation_moderate'] == 'Moderate-Heavy-(>2/3)' else 'Insignificant',
    axis=1
)

# Define a Sunburst chart to visualize hierarchical relationships between damage categories and risk factors
fig = px.sunburst(df, 
                  path=['Damage Category', 'damage_grade', 'has_geotechnical_risk'],  # Hierarchical path
                  title="Building Damage and Geotechnical Risk Analysis")

# Customize the layout for better visualization
fig.update_layout(margin={"t": 0, "l": 0, "r": 0, "b": 0}, 
                  title_font_size=24)

# Show the plot
fig.show()

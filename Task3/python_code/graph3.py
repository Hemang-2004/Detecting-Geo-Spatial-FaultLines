import pandas as pd
import plotly.graph_objects as go

# Load the dataset
data_path = '/home/vasu/Documents/A3/csv_files/csv_household_earthquake_impact.csv'
data = pd.read_csv(data_path)

# Handle missing values by filling with 0 (or you can use .dropna() to remove rows with missing values)
data = data.fillna(0)

# Select relevant columns for Waterfall chart
data_for_waterfall = data[['count_death_last_12_months', 
                           'count_injury_loss_last_12_months', 
                           'is_recipient_rahat_15k', 
                           'is_recipient_rahat_10k', 
                           'is_recipient_rahat_200k', 
                           'is_recipient_rahat_social_security_3k', 
                           'is_recipient_rahat_none']]

# Reshape data to calculate the cumulative impact of Rahat aid
# Create a new column that combines all Rahat receipt types into a single category
data_for_waterfall.loc[:, 'Rahat Type'] = 'No Rahat'  # Use .loc to avoid SettingWithCopyWarning

# Define different types of aid receipt
data_for_waterfall.loc[data_for_waterfall['is_recipient_rahat_15k'] == 1, 'Rahat Type'] = 'Rahat 15k'
data_for_waterfall.loc[data_for_waterfall['is_recipient_rahat_10k'] == 1, 'Rahat Type'] = 'Rahat 10k'
data_for_waterfall.loc[data_for_waterfall['is_recipient_rahat_200k'] == 1, 'Rahat Type'] = 'Rahat 200k'
data_for_waterfall.loc[data_for_waterfall['is_recipient_rahat_social_security_3k'] == 1, 'Rahat Type'] = 'Rahat Social Security 3k'

# Aggregate data to get total deaths and injuries for each Rahat Type
aggregated_data = data_for_waterfall.groupby('Rahat Type').agg({
    'count_death_last_12_months': 'sum',
    'count_injury_loss_last_12_months': 'sum'
}).reset_index()

# Create a Waterfall chart for Death Counts across different Rahat types using plotly.graph_objects
fig_deaths = go.Figure(go.Waterfall(
    x=aggregated_data['Rahat Type'],
    y=aggregated_data['count_death_last_12_months'],
    measure=['relative'] * len(aggregated_data),  # Each step is relative to the previous one
    text=aggregated_data['count_death_last_12_months'].apply(lambda x: f'{x} deaths'),
    base=0
))

fig_deaths.update_layout(
    title="Waterfall Chart: Death Counts Across Rahat Aid Types",
    xaxis_title="Rahat Aid Type",
    yaxis_title="Count of Deaths"
)
fig_deaths.show()

# Create a Waterfall chart for Injury Counts across different Rahat types using plotly.graph_objects
fig_injuries = go.Figure(go.Waterfall(
    x=aggregated_data['Rahat Type'],
    y=aggregated_data['count_injury_loss_last_12_months'],
    measure=['relative'] * len(aggregated_data),  # Each step is relative to the previous one
    text=aggregated_data['count_injury_loss_last_12_months'].apply(lambda x: f'{x} injuries'),
    base=0
))

fig_injuries.update_layout(
    title="Waterfall Chart: Injury Counts Across Rahat Aid Types",
    xaxis_title="Rahat Aid Type",
    yaxis_title="Count of Injuries"
)
fig_injuries.show()

import pandas as pd
import matplotlib.pyplot as plt
import squarify

# Load the dataset
data_path = '/home/vasu/Documents/A3/csv_files/csv_household_earthquake_impact.csv'
data = pd.read_csv(data_path, low_memory=False)

# Fill missing values for clarity (use 'Unknown' for missing data)
data = data.fillna('Unknown')

# Columns related to `rahat` (relief distribution)
rahat_columns = [
    'is_recipient_rahat_15k',
    'is_recipient_rahat_10k',
    'is_recipient_rahat_200k',
    'is_recipient_rahat_social_security_3k',
    'is_recipient_rahat_none',
    'is_ineligible_rahat'
]

# Filter data for rahat analysis
rahat_data = data[rahat_columns]

# Convert the 'Yes'/'No' values to 1 and 0 for easier handling
rahat_data = rahat_data.applymap(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else x))

# Create a dictionary to hold the count of recipients for each relief type
recipient_counts = {col: rahat_data[col].sum() for col in rahat_columns}

# Filter out any zero counts
recipient_counts = {key: value for key, value in recipient_counts.items() if value > 0}

# If there are no non-zero counts, print a message and exit
if not recipient_counts:
    print("No non-zero recipient data found.")
else:
    # Create the labels and values for the tree map
    labels = [f"{key}: {value}" for key, value in recipient_counts.items()]
    values = list(recipient_counts.values())

    # Create the tree map
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=values, label=labels, alpha=0.7, color=['#FF6347', '#FFD700', '#3CB371', '#20B2AA', '#9370DB', '#8A2BE2'])

    # Add title and display the tree map
    plt.title("Tree Map: Rahat Relief Distribution", fontsize=16)
    plt.axis('off')  # Turn off the axis for better visualization

    # Save the figure
    output_path = '/home/vasu/Documents/A3/images/fig46.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Tree map saved as: {output_path}")

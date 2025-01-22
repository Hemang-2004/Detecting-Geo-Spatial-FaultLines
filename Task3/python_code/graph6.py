import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace with your file path)
data = pd.read_csv("/home/vasu/Documents/A3/csv_files/csv_building_damage_assessment.csv")

# List of columns to plot (add or remove columns as needed)
columns_to_plot = [
    "damage_overall_collapse",
    "damage_overall_leaning",
    "damage_foundation_severe",
    "damage_roof_severe",
    "damage_corner_separation_severe",
    # Add more columns as needed
]

# Select an independent variable (e.g., population or building area, etc.)
independent_variable = "district_id"  # Replace with the appropriate column name

# Preprocess data to count occurrences of values for a multi-layered donut chart
for column in columns_to_plot:
    # Count occurrences of values for the column and independent variable combined
    grouped_data = data.groupby([independent_variable, column]).size().unstack(fill_value=0)

    if grouped_data.shape[1] > 1:  # Ensure there are multiple categories to visualize
        plt.figure(figsize=(8, 8))
        # Create the outer layer of the donut chart
        outer_counts = grouped_data.sum(axis=1)
        plt.pie(outer_counts, labels=outer_counts.index, radius=1, autopct="%1.1f%%", startangle=90, 
                wedgeprops={"width": 0.3, "edgecolor": "w"}, labeldistance=1.1)

        # Create the inner layer of the donut chart without labels
        inner_counts = grouped_data.stack()
        plt.pie(inner_counts, radius=0.7, autopct="%1.1f%%", startangle=90, 
                wedgeprops={"width": 0.3, "edgecolor": "w"}, labeldistance=0.6)

        plt.title(f"Multi-Layered Donut Chart for {column} and {independent_variable}")
        plt.tight_layout()
        # Save the plot as a file
        plt.savefig(f"/home/vasu/Documents/A3/donut_chart_{column}.png")
        plt.show()
    else:
        print(f"Not enough data to plot {column} with {independent_variable}.")

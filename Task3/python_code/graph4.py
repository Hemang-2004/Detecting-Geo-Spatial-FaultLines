import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/home/vasu/Documents/A3/csv_files/csv_building_damage_assessment.csv')

# Select the columns you want to include in the node-link diagram
columns_of_interest = ['damage_grade', 'has_geotechnical_risk', 'has_damage_foundation', 'has_damage_roof',
                       'has_damage_corner_separation', 'has_damage_diagonal_cracking', 'has_damage_in_plane_failure']

# Create a graph to represent the relationships
G = nx.Graph()

# Add nodes and edges based on correlation or relationships
for col1 in columns_of_interest:
    for col2 in columns_of_interest:
        if col1 != col2:  # Avoid self-loops
            G.add_edge(col1, col2)

# Create a radial layout for the node-link diagram
pos = nx.spring_layout(G, seed=42, k=0.3)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
plt.title('Node-Link Diagram: Seismic Hazard Relationships')
plt.axis('off')  # Hide axes

# Save the plot
output_path = '/home/vasu/Documents/A3/img_hemang/img1.jpeg'
plt.savefig(output_path)
plt.close()

output_path
import networkx as nx
import matplotlib.pyplot as plt
from graph_manager import load_graph

def export_graph_to_file():
    print("📂 Loading Knowledge Graph from 'scheme_graph.pkl'...")
    G = load_graph()
    
    print(f"📊 Total Nodes in Graph: {G.number_of_nodes()}")
    print("🎨 Generating visual graph layout (exporting to PDF/PNG)...")
    
    plt.figure(figsize=(14, 10))
    
    # Use spring layout for clean spacing of nodes and edges
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw departments, schemes, and beneficiaries with different colors or sizes
    node_colors = []
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        if ntype == "Department":
            node_colors.append("#ff7f0e")  # Orange for Departments
        elif ntype == "Scheme":
            node_colors.append("#1f77b4")  # Blue for Schemes
        else:
            node_colors.append("#2ca02c")  # Green for Beneficiaries
            
    # Draw graph elements
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color="gray", arrows=True)
    
    # Label nodes (only showing a subset or shortened labels to prevent text overlapping)
    labels = {node: (node[:25] + "..." if len(node) > 25 else node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="sans-serif")
    
    plt.title("Tamil Nadu Government Schemes Knowledge Graph (GraphRAG)", fontsize=14, fontweight="bold")
    plt.axis("off")
    
    # Save as high-resolution PNG and PDF
    plt.tight_layout()
    plt.savefig("tn_schemes_knowledge_graph.png", dpi=300, bbox_inches="tight")
    plt.savefig("tn_schemes_knowledge_graph.pdf", bbox_inches="tight")
    
    print("✅ Successfully exported!")
    print(" - Saved as: tn_schemes_knowledge_graph.png")
    print(" - Saved as: tn_schemes_knowledge_graph.pdf")
    
    plt.show()

if __name__ == "__main__":
    export_graph_to_file()
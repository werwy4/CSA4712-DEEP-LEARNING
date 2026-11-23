# Experiment No: 61
# Title: Bayesian Network Conditional Probability Tables
# Aim: Construct a Bayesian network and define CPTs for machine maintenance (overheating, cooling failure, vibration, shutdown).
# Libraries: pgmpy, NetworkX, Pandas, Matplotlib

# Install: pip install pgmpy networkx pandas matplotlib

# PROGRAM
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    from pgmpy.models import BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianModel as BayesianNetwork

from pgmpy.factors.discrete import TabularCPD

def main():
    print("=" * 65)
    print("  LAB 61: BAYESIAN NETWORK CONDITIONAL PROBABILITY TABLES")
    print("=" * 65)

    # Step 1: Define Network Architecture
    # Directed dependencies:
    # Overheating -> CoolingFailure
    # Overheating -> Shutdown
    # Vibration -> Shutdown
    # CoolingFailure -> Shutdown
    model = BayesianNetwork([
        ('Overheating', 'CoolingFailure'),
        ('Overheating', 'Shutdown'),
        ('Vibration', 'Shutdown'),
        ('CoolingFailure', 'Shutdown')
    ])

    print("\n[+] Defined Bayesian Network Edges:")
    for edge in model.edges():
        print(f"    {edge[0]} -> {edge[1]}")

    # Step 2: Define Conditional Probability Distributions (CPDs)
    # State mapping: 0 = Normal / No, 1 = Fault / Yes

    # Prior for Overheating: P(Overheating)
    # P(Overheating=0) = 0.85, P(Overheating=1) = 0.15
    cpd_overheating = TabularCPD(
        variable='Overheating',
        variable_card=2,
        values=[[0.85], [0.15]],
        state_names={'Overheating': ['No', 'Yes']}
    )

    # Prior for Vibration: P(Vibration)
    # P(Vibration=0) = 0.80, P(Vibration=1) = 0.20
    cpd_vibration = TabularCPD(
        variable='Vibration',
        variable_card=2,
        values=[[0.80], [0.20]],
        state_names={'Vibration': ['No', 'Yes']}
    )

    # Conditional for CoolingFailure given Overheating: P(CoolingFailure | Overheating)
    # If Overheating=No: P(CoolingFail=No)=0.95, P(CoolingFail=Yes)=0.05
    # If Overheating=Yes: P(CoolingFail=No)=0.40, P(CoolingFail=Yes)=0.60
    cpd_cooling = TabularCPD(
        variable='CoolingFailure',
        variable_card=2,
        values=[
            [0.95, 0.40],  # No
            [0.05, 0.60]   # Yes
        ],
        evidence=['Overheating'],
        evidence_card=[2],
        state_names={
            'CoolingFailure': ['No', 'Yes'],
            'Overheating': ['No', 'Yes']
        }
    )

    # Conditional for Shutdown given CoolingFailure, Overheating, Vibration
    # Evidence order: CoolingFailure, Overheating, Vibration (2 x 2 x 2 = 8 combinations)
    # Probability of Shutdown=No (row 0) and Shutdown=Yes (row 1)
    # Combinations (CoolingFailure, Overheating, Vibration):
    # (No,No,No), (No,No,Yes), (No,Yes,No), (No,Yes,Yes),
    # (Yes,No,No), (Yes,No,Yes), (Yes,Yes,No), (Yes,Yes,Yes)
    shutdown_probs = [
        [0.99, 0.60, 0.50, 0.15, 0.40, 0.10, 0.05, 0.01],  # Shutdown = No
        [0.01, 0.40, 0.50, 0.85, 0.60, 0.90, 0.95, 0.99]   # Shutdown = Yes
    ]

    cpd_shutdown = TabularCPD(
        variable='Shutdown',
        variable_card=2,
        values=shutdown_probs,
        evidence=['CoolingFailure', 'Overheating', 'Vibration'],
        evidence_card=[2, 2, 2],
        state_names={
            'Shutdown': ['No', 'Yes'],
            'CoolingFailure': ['No', 'Yes'],
            'Overheating': ['No', 'Yes'],
            'Vibration': ['No', 'Yes']
        }
    )

    # Step 3: Add CPDs to the Model and Validate
    model.add_cpds(cpd_overheating, cpd_vibration, cpd_cooling, cpd_shutdown)
    is_valid = model.check_model()
    print(f"\n[+] Model Validation Check: {'VALID' if is_valid else 'INVALID'}")

    # Display CPDs
    print("\n--- Conditional Probability Distributions (CPDs) ---")
    print("\n1. P(Overheating):")
    print(model.get_cpds('Overheating'))
    print("\n2. P(Vibration):")
    print(model.get_cpds('Vibration'))
    print("\n3. P(CoolingFailure | Overheating):")
    print(model.get_cpds('CoolingFailure'))
    print("\n4. P(Shutdown | CoolingFailure, Overheating, Vibration):")
    print(model.get_cpds('Shutdown'))

    # Step 4: Identify Conditional Independencies
    print("\n--- Conditional Independencies in the Network ---")
    independencies = model.get_independencies()
    print(independencies)

    # Step 5: Visualise Directed Graph
    plt.figure(figsize=(8, 6))
    pos = {
        'Overheating': (0.2, 0.8),
        'Vibration': (0.8, 0.8),
        'CoolingFailure': (0.2, 0.4),
        'Shutdown': (0.5, 0.1)
    }

    nx_graph = nx.DiGraph(model.edges())
    nx.draw_networkx_nodes(nx_graph, pos, node_size=3000, node_color='#87CEEB', edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold', font_family='sans-serif')
    nx.draw_networkx_edges(nx_graph, pos, arrows=True, arrowsize=25, edge_color='#2E4053', width=2,
                           connectionstyle="arc3,rad=0.05")

    plt.title("Bayesian Network: Machine Maintenance System", fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("lab61_bayesian_network.png", dpi=300)
    print("\n[+] Directed graph saved as 'lab61_bayesian_network.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 61: BAYESIAN NETWORK CONDITIONAL PROBABILITY TABLES
# =================================================================
#
# [+] Defined Bayesian Network Edges:
#     Overheating -> CoolingFailure
#     Overheating -> Shutdown
#     Vibration -> Shutdown
#     CoolingFailure -> Shutdown
#
# [+] Model Validation Check: VALID
#
# --- Conditional Probability Distributions (CPDs) ---
# 1. P(Overheating): TabularCPD with states No: 0.85, Yes: 0.15
# 2. P(Vibration): TabularCPD with states No: 0.80, Yes: 0.20
# 3. P(CoolingFailure | Overheating): 2x2 CPT conditioned on Overheating
# 4. P(Shutdown | CoolingFailure, Overheating, Vibration): 2x8 CPT
#
# --- Conditional Independencies in the Network ---
# (CoolingFailure \perp Vibration | Overheating)
# (Vibration \perp CoolingFailure, Overheating)
#
# [+] Directed graph saved as 'lab61_bayesian_network.png'

# RESULT
# Successfully constructed a Directed Acyclic Graph (DAG) for a machine maintenance system,
# configured its Conditional Probability Tables (CPTs) representing causal dependencies,
# validated the probabilistic consistency of the network, extracted conditional independencies,
# and visualised the Bayesian network structure using NetworkX and Matplotlib.

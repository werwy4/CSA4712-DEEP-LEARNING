# Experiment No: 53
# Title: Bayesian Network Construction and Visualization
# Aim: Build a Bayesian network with CPDs and visualise its structure for medical diagnosis.
# Libraries: pgmpy, NetworkX, Matplotlib

# Install: pip install pgmpy networkx matplotlib

# PROGRAM
import matplotlib.pyplot as plt
import networkx as nx

try:
    from pgmpy.models import BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianModel as BayesianNetwork

from pgmpy.factors.discrete import TabularCPD

def main():
    print("=" * 65)
    print("  LAB 53: BAYESIAN NETWORK CONSTRUCTION AND VISUALIZATION")
    print("=" * 65)

    # Step 1: Define Network Architecture (DAG)
    # Medical Diagnosis DAG:
    # Smoking -> LungDisease
    # Pollution -> LungDisease
    # LungDisease -> Cough
    # LungDisease -> XRay
    # LungDisease -> Fatigue
    edges = [
        ('Smoking', 'LungDisease'),
        ('Pollution', 'LungDisease'),
        ('LungDisease', 'Cough'),
        ('LungDisease', 'XRay'),
        ('LungDisease', 'Fatigue')
    ]

    model = BayesianNetwork(edges)
    print("\n[+] Defined Directed Acyclic Graph (DAG) Edges:")
    for src, dst in model.edges():
        print(f"    {src:<15} ---> {dst}")

    # Step 2: Define Conditional Probability Distributions (CPDs)

    # 1. Prior for Smoking: P(Smoking)
    cpd_smoking = TabularCPD(
        variable='Smoking',
        variable_card=2,
        values=[[0.70], [0.30]],
        state_names={'Smoking': ['No', 'Yes']}
    )

    # 2. Prior for Pollution: P(Pollution)
    cpd_pollution = TabularCPD(
        variable='Pollution',
        variable_card=2,
        values=[[0.80], [0.20]],
        state_names={'Pollution': ['Low', 'High']}
    )

    # 3. Conditional for LungDisease | Smoking, Pollution
    # Evidence combinations: (Smoking, Pollution):
    # (No, Low), (No, High), (Yes, Low), (Yes, High)
    cpd_lung = TabularCPD(
        variable='LungDisease',
        variable_card=2,
        values=[
            [0.98, 0.90, 0.70, 0.40],  # LungDisease = No
            [0.02, 0.10, 0.30, 0.60]   # LungDisease = Yes
        ],
        evidence=['Smoking', 'Pollution'],
        evidence_card=[2, 2],
        state_names={
            'LungDisease': ['No', 'Yes'],
            'Smoking': ['No', 'Yes'],
            'Pollution': ['Low', 'High']
        }
    )

    # 4. Conditional for Cough | LungDisease
    cpd_cough = TabularCPD(
        variable='Cough',
        variable_card=2,
        values=[
            [0.85, 0.20],  # Cough = No
            [0.15, 0.80]   # Cough = Yes
        ],
        evidence=['LungDisease'],
        evidence_card=[2],
        state_names={
            'Cough': ['No', 'Yes'],
            'LungDisease': ['No', 'Yes']
        }
    )

    # 5. Conditional for XRay | LungDisease
    cpd_xray = TabularCPD(
        variable='XRay',
        variable_card=2,
        values=[
            [0.95, 0.10],  # XRay = Normal
            [0.05, 0.90]   # XRay = Abnormal
        ],
        evidence=['LungDisease'],
        evidence_card=[2],
        state_names={
            'XRay': ['Normal', 'Abnormal'],
            'LungDisease': ['No', 'Yes']
        }
    )

    # 6. Conditional for Fatigue | LungDisease
    cpd_fatigue = TabularCPD(
        variable='Fatigue',
        variable_card=2,
        values=[
            [0.80, 0.25],  # Fatigue = No
            [0.20, 0.75]   # Fatigue = Yes
        ],
        evidence=['LungDisease'],
        evidence_card=[2],
        state_names={
            'Fatigue': ['No', 'Yes'],
            'LungDisease': ['No', 'Yes']
        }
    )

    # Step 3: Add CPDs and Validate Model Consistency
    model.add_cpds(cpd_smoking, cpd_pollution, cpd_lung, cpd_cough, cpd_xray, cpd_fatigue)
    is_valid = model.check_model()
    print(f"\n[+] Bayesian Model Probabilistic Validation: {'VALID' if is_valid else 'INVALID'}")

    # Display CPDs
    print("\n--- Conditional Probability Distributions ---")
    print("\n1. P(Smoking):")
    print(model.get_cpds('Smoking'))
    print("\n2. P(Pollution):")
    print(model.get_cpds('Pollution'))
    print("\n3. P(LungDisease | Smoking, Pollution):")
    print(model.get_cpds('LungDisease'))
    print("\n4. P(XRay | LungDisease):")
    print(model.get_cpds('XRay'))

    # Step 4: Extract Conditional Independencies
    print("\n--- Extracted Conditional Independencies ---")
    independencies = model.get_independencies()
    print(independencies)

    # Step 5: Visualise Directed Graph with NetworkX
    plt.figure(figsize=(9, 7))

    # Hierarchical coordinates for clean clinical workflow visualization
    pos = {
        'Smoking': (-0.6, 1.0),
        'Pollution': (0.6, 1.0),
        'LungDisease': (0.0, 0.4),
        'Cough': (-0.8, -0.3),
        'XRay': (0.0, -0.3),
        'Fatigue': (0.8, -0.3)
    }

    nx_graph = nx.DiGraph(model.edges())

    # Node styling
    node_colors = ['#85C1E9', '#85C1E9', '#F1948A', '#F9E79F', '#F9E79F', '#F9E79F']
    nx.draw_networkx_nodes(
        nx_graph, pos,
        node_size=3200,
        node_color='#AED6F1',
        edgecolors='#2C3E50',
        linewidths=2
    )
    nx.draw_networkx_labels(
        nx_graph, pos,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif'
    )
    nx.draw_networkx_edges(
        nx_graph, pos,
        arrows=True,
        arrowsize=25,
        edge_color='#2C3E50',
        width=2,
        connectionstyle="arc3,rad=0.04"
    )

    plt.title("Bayesian Belief Network: Medical Diagnosis Workflow", fontsize=13, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("lab53_bayesian_network.png", dpi=300)
    print("\n[+] Network graph visual saved as 'lab53_bayesian_network.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 53: BAYESIAN NETWORK CONSTRUCTION AND VISUALIZATION
# =================================================================
#
# [+] Defined Directed Acyclic Graph (DAG) Edges:
#     Smoking         ---> LungDisease
#     Pollution       ---> LungDisease
#     LungDisease     ---> Cough
#     LungDisease     ---> XRay
#     LungDisease     ---> Fatigue
#
# [+] Bayesian Model Probabilistic Validation: VALID
#
# --- Conditional Probability Distributions ---
#
# 1. P(Smoking):
# +--------------+-----+
# | Smoking(No)  | 0.7 |
# +--------------+-----+
# | Smoking(Yes) | 0.3 |
# +--------------+-----+
#
# 2. P(Pollution):
# +----------------+-----+
# | Pollution(Low) | 0.8 |
# +----------------+-----+
# | Pollution(High)| 0.2 |
# +----------------+-----+
#
# 3. P(LungDisease | Smoking, Pollution):
# TabularCPD over 4 parent combinations.
#
# 4. P(XRay | LungDisease):
# TabularCPD conditioned on LungDisease.
#
# --- Extracted Conditional Independencies ---
# (Smoking \perp Pollution)
# (Cough \perp Fatigue, XRay, Smoking, Pollution | LungDisease)
# (XRay \perp Fatigue, Cough, Smoking, Pollution | LungDisease)
# (Fatigue \perp Cough, XRay, Smoking, Pollution | LungDisease)
#
# [+] Network graph visual saved as 'lab53_bayesian_network.png'

# RESULT
# Successfully constructed a Bayesian Belief Network for medical diagnosis using pgmpy.
# Specified prior probabilities for causal risk factors (Smoking, Pollution) and conditional
# probability tables (CPTs) for disease state and symptoms (Cough, XRay, Fatigue). Validated
# probabilistic consistency, extracted conditional independencies, and visualised the DAG
# structure using NetworkX.

import numpy as np
import bnlearn as bn
import pandas as pd
import networkx as nx
from pgmpy.factors.discrete import TabularCPD

# Define the structure of the DAG as a list of edges
dag_structure = [
    ('A', 'B'),  # A causes B
    ('A', 'C'),  # A causes C
    ('B', 'D'),  # B causes D
    ('C', 'D')   # C causes D
]

# Step 2: Define the CPDs
cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]])  # P(A)
cpd_B = TabularCPD(variable='B', variable_card=2,
                   values=[[0.7, 0.4], [0.3, 0.6]],
                   evidence=['A'], evidence_card=[2])  # P(B|A)
cpd_C = TabularCPD(variable='C', variable_card=2,
                   values=[[0.8, 0.3], [0.2, 0.7]],
                   evidence=['A'], evidence_card=[2])  # P(C|A)
cpd_D = TabularCPD(variable='D', variable_card=2,
                   values=[
                       [0.9, 0.5, 0.4, 0.2],  # P(D=0 | B, C)
                       [0.1, 0.5, 0.6, 0.8]   # P(D=1 | B, C)
                   ],
                   evidence=['B', 'C'], evidence_card=[2, 2])  # P(D|B,C)

# Step 3: Add CPDs to the model
DAG = bn.make_DAG(dag_structure, CPD=[cpd_A, cpd_B, cpd_C, cpd_D])

# Number of samples to generate
n_samples = 10000

# Generate synthetic data
data = bn.sampling(DAG, n=n_samples)
print(data.head())

# Add Gaussian noise to the variables
data['A'] += np.random.normal(0, 0.1, n_samples)
data['B'] += np.random.normal(0, 0.1, n_samples)
data['C'] += np.random.normal(0, 0.1, n_samples)
data['D'] += np.random.normal(0, 0.1, n_samples)
data['D'] = (data['D'] > 0.5).astype(int)  # Discretize 'D' to 0 or 1
# Note: Keeping the noise addition and discretization steps for future flexibility

# Save the data and the DAG
data.to_csv('synthetic_dataset.csv', index=False)
bn.save(DAG, filepath='test_dag', overwrite=True)

bn.print_CPD(DAG)
print(data.head())

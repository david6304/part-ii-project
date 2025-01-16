import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rna_seq_path = 'data/real/5121262/Data from RNASeq Camk2CreDyrk1a-Dyrk1a.xlsx'
rna_seq_data = pd.read_excel(rna_seq_path, sheet_name=0)

# Create a volcano plot from RNA-seq data
ko_cols = ['Dyrk1a_M76_ko', 'Dyrk1a_M75_ko']
wt_cols = ['Dyrk1a_M96', 'Dyrk1a_M95']
rna_seq_data['fold change'] = rna_seq_data[ko_cols].mean(axis=1) - rna_seq_data[wt_cols].mean(axis=1)
p_value_col = 'Pval DEseq'
rna_seq_data['-log10(p-value)'] = -np.log10(rna_seq_data[p_value_col])

plt.figure(figsize=(10, 6))
plt.scatter(rna_seq_data['fold change'], rna_seq_data['-log10(p-value)'], alpha=0.5)
plt.title('Volcano Plot')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10(p-value)')
plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
plt.axvline(x=1, color='r', linestyle='--')
plt.axvline(x=-1, color='r', linestyle='--')
plt.show()


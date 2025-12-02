import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the CSV data
df = pd.read_csv('combined_matrix_150.csv')

# Replace 'Na' with np.nan
df.replace('Na', np.nan, inplace=True)

# Set 'name' as the index if it exists
if 'name' in df.columns:
    df.set_index('name', inplace=True)

# Transpose the matrix (make each sample a row)
df_transposed = df.T.astype(float)

# Perform multiple imputation (handle missing values)
# Since random_state=0, the results should match the original code
imp = IterativeImputer(random_state=0, max_iter=10)
data_imputed = imp.fit_transform(df_transposed)

# Restore the DataFrame structure
df_imputed = pd.DataFrame(data_imputed, index=df_transposed.index, columns=df_transposed.columns)

# Standardization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_imputed)

# Revised group classification function
def get_group_label_fixed(name):
    match = re.search(r'(28514744|32144372|33456364|34737022|34887646|35116043)(?:_(\w+))?$', name)
    if match:
        group_id, sample_id = match.groups()
        return group_id, sample_id if sample_id else ''
    return 'Other', name

# Retrieve groups and labels
group_labels_fixed = [get_group_label_fixed(name) for name in df_imputed.index]
groups_fixed = [label[0] for label in group_labels_fixed]
labels_fixed = [label[1] for label in group_labels_fixed]

unique_groups_fixed = sorted(set(groups_fixed))
# Reviewer precaution: Use a color-vision-friendly palette
palette_fixed = sns.color_palette('colorblind', len(unique_groups_fixed)) 
group_color_dict_fixed = {group: palette_fixed[i] for i, group in enumerate(unique_groups_fixed)}

# Run PCA
pca_fixed = PCA(n_components=2)
pca_result_fixed = pca_fixed.fit_transform(data_scaled)
pca_df_fixed = pd.DataFrame(pca_result_fixed, columns=['PC1', 'PC2'], index=df_imputed.index)

explained_variance = pca_fixed.explained_variance_ratio_
pc1_var = explained_variance[0] * 100
pc2_var = explained_variance[1] * 100

# t-SNE
# ※The parameters are kept exactly the same as in PCA_tSNE_3.py
tsne_fixed = TSNE(n_components=2, random_state=0, perplexity=3, n_iter=500, init='pca', learning_rate='auto')
tsne_result_fixed = tsne_fixed.fit_transform(data_scaled)
tsne_df_fixed = pd.DataFrame(tsne_result_fixed, columns=['tSNE1', 'tSNE2'], index=df_imputed.index)

# Create plots (PCA and t-SNE)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# PCA plot
for i, sample_name in enumerate(pca_df_fixed.index):
    x, y = pca_df_fixed.loc[sample_name]
    group = groups_fixed[i]
    axes[0].scatter(x, y, color=group_color_dict_fixed[group], s=120)
    axes[0].text(x + 0.03, y, labels_fixed[i], fontsize=11, fontweight='bold')

axes[0].set_title('PCA Scatter Plot (Grouped Samples)', fontsize=14)

axes[0].set_xlabel(f'Principal Component 1 ({pc1_var:.1f}%)')
axes[0].set_ylabel(f'Principal Component 2 ({pc2_var:.1f}%)')
axes[0].grid(True)

# t-SNE plot
for i, sample_name in enumerate(tsne_df_fixed.index):
    x, y = tsne_df_fixed.loc[sample_name]
    group = groups_fixed[i]
    axes[1].scatter(x, y, color=group_color_dict_fixed[group], s=120)
    axes[1].text(x + 0.03, y, labels_fixed[i], fontsize=11, fontweight='bold')

axes[1].set_title('t-SNE Scatter Plot (Grouped Samples)', fontsize=14)
axes[1].set_xlabel('t-SNE Dimension 1') # Do not include % in t-SNE
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].grid(True)

# Display the legend
handles_fixed = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=group_color_dict_fixed[group], markersize=12)
                for group in unique_groups_fixed]

# Place the legend in the lower-left corner of each plot
for ax in axes:
    ax.legend(handles_fixed, unique_groups_fixed, 
             loc='lower left',
             title='Groups',
             fontsize=10,
             title_fontsize=12,
             framealpha=0.9,
             edgecolor='black')

# Adjust margins and save
plt.tight_layout()
output_filename = 'pca_tsne_final_revised.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Image saved: {output_filename}")
plt.show()

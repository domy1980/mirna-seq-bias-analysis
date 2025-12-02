
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# CSV loading
df = pd.read_csv('common_matrix.csv') # combined_matrix_150.csv

# 'Na' -> np.nan replace
df.replace('Na', np.nan, inplace=True)

# Perform imputation only on numeric columns (exclude 'name')
if 'name' in df.columns:
    names = df['name']
    df_numeric = df.drop(columns=['name']).astype(float)
else:
    df_numeric = df.astype(float)

# Perform multiple imputation
imp = IterativeImputer(random_state=0, max_iter=10)
df_imputed = imp.fit_transform(df_numeric)

# Return to the data frame after imputation
df_imputed = pd.DataFrame(df_imputed, columns=df_numeric.columns)

# Calculate the Spearman correlation coefficient
spearman_corr = df_imputed.corr(method='spearman')

# Definition of vivid custom colors
vivid_colors = [
    (0.0, '#0000FF'),  # Vivid blue
    (0.000001, '#0000FF'),  # Keep the color blue until it slightly exceeds zero
    (0.5, '#FFFFFF'),  # White
    (1.0, '#FF0000')   # Vivid red
]
vivid_custom_cmap = LinearSegmentedColormap.from_list('vivid_custom_heatmap', vivid_colors)

# Display a Spearman correlation heatmap (apply vivid custom colors)
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, cmap=vivid_custom_cmap, center=0.5, annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title('Spearman Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# Save the imputed data as needed
df_imputed.to_csv('imputed_combined_matrix.csv', index=False)

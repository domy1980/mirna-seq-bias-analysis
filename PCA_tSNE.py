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

# ---------------------------------------------------------
# ベースは PCA_tSNE_3.py をそのまま使用
# ---------------------------------------------------------

# CSVデータを読み込む
df = pd.read_csv('combined_matrix_150.csv')

# 'Na'をnp.nanに置換
df.replace('Na', np.nan, inplace=True)

# 'name'があればindexとして設定
if 'name' in df.columns:
    df.set_index('name', inplace=True)

# 行列を転置（各サンプルを行にする）
df_transposed = df.T.astype(float)

# 多重補完を実施（欠損値の処理）
# ※random_state=0 なので、元のコードと同じ結果になるはずです
imp = IterativeImputer(random_state=0, max_iter=10)
data_imputed = imp.fit_transform(df_transposed)

# DataFrame構造に戻す
df_imputed = pd.DataFrame(data_imputed, index=df_transposed.index, columns=df_transposed.columns)

# 標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_imputed)

# 修正版グループ分け関数
def get_group_label_fixed(name):
    match = re.search(r'(28514744|32144372|33456364|34737022|34887646|35116043)(?:_(\w+))?$', name)
    if match:
        group_id, sample_id = match.groups()
        return group_id, sample_id if sample_id else ''
    return 'Other', name

# グループとラベル取得
group_labels_fixed = [get_group_label_fixed(name) for name in df_imputed.index]
groups_fixed = [label[0] for label in group_labels_fixed]
labels_fixed = [label[1] for label in group_labels_fixed]

# ---------------------------------------------------------
# 変更点1: 色パレットを 'tab10' から 'colorblind' に変更
# ---------------------------------------------------------
unique_groups_fixed = sorted(set(groups_fixed))
# Reviewer対策: 色覚バリアフリーパレットを使用
palette_fixed = sns.color_palette('colorblind', len(unique_groups_fixed)) 
group_color_dict_fixed = {group: palette_fixed[i] for i, group in enumerate(unique_groups_fixed)}

# PCA実行
pca_fixed = PCA(n_components=2)
pca_result_fixed = pca_fixed.fit_transform(data_scaled)
pca_df_fixed = pd.DataFrame(pca_result_fixed, columns=['PC1', 'PC2'], index=df_imputed.index)

# ---------------------------------------------------------
# 変更点2: PCAの分散（寄与率）を計算
# ---------------------------------------------------------
explained_variance = pca_fixed.explained_variance_ratio_
pc1_var = explained_variance[0] * 100
pc2_var = explained_variance[1] * 100

# t-SNE実行
# ※パラメータは PCA_tSNE_3.py と完全に同じにしています
tsne_fixed = TSNE(n_components=2, random_state=0, perplexity=3, n_iter=500, init='pca', learning_rate='auto')
tsne_result_fixed = tsne_fixed.fit_transform(data_scaled)
tsne_df_fixed = pd.DataFrame(tsne_result_fixed, columns=['tSNE1', 'tSNE2'], index=df_imputed.index)

# プロット作成（PCAとt-SNE）
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# PCAプロット
for i, sample_name in enumerate(pca_df_fixed.index):
    x, y = pca_df_fixed.loc[sample_name]
    group = groups_fixed[i]
    axes[0].scatter(x, y, color=group_color_dict_fixed[group], s=120)
    axes[0].text(x + 0.03, y, labels_fixed[i], fontsize=11, fontweight='bold')

axes[0].set_title('PCA Scatter Plot (Grouped Samples)', fontsize=14)

# ---------------------------------------------------------
# 変更点3: PCA軸ラベルに分散（%）を表示
# ---------------------------------------------------------
axes[0].set_xlabel(f'Principal Component 1 ({pc1_var:.1f}%)')
axes[0].set_ylabel(f'Principal Component 2 ({pc2_var:.1f}%)')
axes[0].grid(True)

# t-SNEプロット
for i, sample_name in enumerate(tsne_df_fixed.index):
    x, y = tsne_df_fixed.loc[sample_name]
    group = groups_fixed[i]
    axes[1].scatter(x, y, color=group_color_dict_fixed[group], s=120)
    axes[1].text(x + 0.03, y, labels_fixed[i], fontsize=11, fontweight='bold')

axes[1].set_title('t-SNE Scatter Plot (Grouped Samples)', fontsize=14)
axes[1].set_xlabel('t-SNE Dimension 1') # t-SNEには%は入れない
axes[1].set_ylabel('t-SNE Dimension 2')
axes[1].grid(True)

# 凡例を表示
handles_fixed = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=group_color_dict_fixed[group], markersize=12)
                for group in unique_groups_fixed]

# 各プロットの左下に凡例を配置
for ax in axes:
    ax.legend(handles_fixed, unique_groups_fixed, 
             loc='lower left',
             title='Groups',
             fontsize=10,
             title_fontsize=12,
             framealpha=0.9,
             edgecolor='black')

# 余白調整して保存
plt.tight_layout()
output_filename = 'pca_tsne_final_revised.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"画像を保存しました: {output_filename}")
plt.show()
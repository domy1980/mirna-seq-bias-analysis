
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# CSVを読み込む（ファイル名をご自身のファイル名に変更）
df = pd.read_csv('common_matrix.csv') # combined_matrix_150.csv

# 'Na'をnp.nanに置換
df.replace('Na', np.nan, inplace=True)

# 数値カラムのみに補完処理を行う（'name'などは除外）
if 'name' in df.columns:
    names = df['name']
    df_numeric = df.drop(columns=['name']).astype(float)
else:
    df_numeric = df.astype(float)

# 多重補完の実施
imp = IterativeImputer(random_state=0, max_iter=10)
df_imputed = imp.fit_transform(df_numeric)

# 補完後のデータフレームに戻す
df_imputed = pd.DataFrame(df_imputed, columns=df_numeric.columns)

# スピアマン相関係数を計算
spearman_corr = df_imputed.corr(method='spearman')

# 鮮やかなカスタムカラーの定義
vivid_colors = [
    (0.0, '#0000FF'),  # 鮮やかな青
    (0.000001, '#0000FF'),  # 微小に0を超えるところまで青固定
    (0.5, '#FFFFFF'),  # 白
    (1.0, '#FF0000')   # 鮮やかな赤
]
vivid_custom_cmap = LinearSegmentedColormap.from_list('vivid_custom_heatmap', vivid_colors)

# スピアマン相関のヒートマップ表示（鮮やかなカスタムカラー適用）
plt.figure(figsize=(12, 10))
sns.heatmap(spearman_corr, cmap=vivid_custom_cmap, center=0.5, annot=True, fmt=".2f", vmin=0, vmax=1)
plt.title('Spearman Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()

# 必要に応じて補完済みデータを保存
df_imputed.to_csv('imputed_combined_matrix.csv', index=False)

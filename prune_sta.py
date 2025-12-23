import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# 1. 載入數據
data = {
    "99b272ae23876c2a554b2b4a5a584bd7cb74d330": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "aaf0de4024c34ec7892d25b49acb2972216e0df0": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "dd6603d528ca9f4f332b5e40a5c717cc2a69a0a6": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "b17680474b0407bf4732db4a6f91a33209837052": [
        0.0,
        0.4785534683614969,
        1.0,
        1.0
    ],
    "3f5f8f71d06408fbe1ca4a8d6dcd811478f79a0e": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "7f22cea795318dab30635f080b5cb72cdfc59b85": [
        0.0,
        0.01675957441329956,
        1.0,
        1.0
    ],
    "e99e57084b8819fdd2b26daa6e735ed2db7ed558": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "777301f951a15f340f19587be055b552b381ee52": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "5cb019e15c6270661ed1ed49b708181710b0a006": [
        0.0,
        0.9166063070297241,
        1.0,
        1.0
    ],
    "3b09681b9550b67b565fc241889234cd57f5e79c": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "d2c539a4e5ce2d9f2906fc1a0678e57200920dd0": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "6441a901ff944fee38d8d1b11d3302ffccb1d5c4": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "1d7425378cbe029537e1ec4b78fb4dfacc39e217": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "2176d948af3c2339735b1e59258d54ad99571fe6": [
        0.0,
        0.10177487134933472,
        1.0,
        1.0
    ],
    "770c762184ab23a162d0700ccc65a773e85696e1": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "ac149c14b326c82c9831c4f724828a93ff1fb424": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "529bb08b209f976db1dd09a228c793290aedfe99": [
        0.0,
        0.3305172473192215,
        1.0,
        1.0
    ],
    "b7aaee5d323fa468366fd31dd0f4c2a59d8348b0": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "fa1b1b4a6a357acfde3bd359c045f946c3fe3b7c": [
        0.0,
        0.5636987164616585,
        1.0,
        1.0
    ],
    "a25956c536b45e8982aff1554bd863348bc0f0fb": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "099fd2da5aa25f4515320b0b6391eec98f5d81b7": [
        0.0,
        0.05497017502784729,
        1.0,
        1.0
    ],
    "f7f712aa026eeeed5d0f3ff0c178b998e8458702": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "8fe1a3a233c40497a4b4d8c71ecc3b27539c8176": [
        0.0,
        0.24835243821144104,
        1.0,
        1.0
    ],
    "c717db50074b8511322347967a970dbd2b1957aa": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "9d8b8bb8a988f74b1e08fa1c4f3f65b50768d5a8": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "1df5793b5155179f70fa2cd86585202f39fe8f4f": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "c0be8ba511b0fcbdac1782ffb0f4b6bc8f474ec2": [
        0.0,
        0.665568932890892,
        1.0,
        1.0
    ],
    "12408d092447a808500a579c3eef46bbcb3c3178": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "e363dd188593cd16717f519aa4b3ea09690f79d7": [
        0.0,
        0.22250735759735107,
        1.0,
        1.0
    ],
    "03b0f113c0255a7257ccf1cf0f2b8e90008e9f9f": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "c2862ccd34a30713aa6be6e6cac9a25350b7b798": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "b01ee2fbc2b805a12313a900d5f49dfe76ae5fd8": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "095af2c2ed3927d532e0acfb363ef9223bfbfdb8": [
        0.0,
        0.14186328649520874,
        1.0,
        1.0
    ],
    "5468539d1e7a6b146bf81136e7fa37cb0dab68e5": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "84bd64f625e8ddc05208228e7c1c421f94934fd2": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "8c15e0f4e4921cd4aaabee122f402585b402376d": [
        0.0,
        0.34554165601730347,
        1.0,
        1.0
    ],
    "61e0e39f51f761f8de8a2ad7ab9c0d90e20bf8d8": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "bdb60b5b3b3d2cc8d294f66332b5df09a908a66e": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "4fb1bd719fdf3e085b41fac44b2ff23bdf2d7250": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "dba28a281cdc1eed895ad5f8927f87945219ec29": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "7cb528fd2f9a1b6bfe767877be5c733c13b1e0f8": [
        0.0,
        0.13474735617637634,
        1.0,
        1.0
    ],
    "c92f12cd0c6f866b2ca1bb6ff86cabec688e21fd": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "8ee1e69d527b57177817300581975971b46bed78": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "abf4a9942fd7581d7afb46465620fe93569854af": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "0578fba88eddfd13cd518aa059223f46a05b6764": [
        0.0,
        0.12666800618171692,
        1.0,
        1.0
    ],
    "6bc196f8d43a22e688bff83585f6766ffd8a301f": [
        0.0,
        0.28323304653167725,
        1.0,
        1.0
    ],
    "da133384164fabafdccf74e2cb1516f0d8b1935c": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "125587b6e8668882b8f4757c439943f1db6b8add": [
        0.0,
        0.0,
        1.0,
        1.0
    ],
    "920356e6793a2178ae773c17d4cb873fcd429927": [
        0.0,
        1.0,
        1.0,
        1.0
    ],
    "91717ce407b296450697d297bd7fbac92df9d5aa": [
        0.0,
        0.3041767030954361,
        1.0,
        1.0
    ]
}

# 2. 轉換為長格式 (Long Format)
rows = []
scales = ['K-3', 'K-2', 'K-1', 'K']
for sample_id, p_values in data.items():
    for i, p_val in enumerate(p_values):
        rows.append({'Scale': scales[i], 'Pruning Ratio': p_val})

df = pd.DataFrame(rows)

# 3. 繪圖設定
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 繪製小提琴圖 (展示 ATP-VAR 的分佈)
ax = sns.violinplot(x='Scale', y='Pruning Ratio', data=df, 
                    inner="quart", color="#a1c9f4", cut=0)

# 4. 疊加 FastVAR 的固定比例點 (作為 Baseline 對比)
fastvar_ratios = [0.4, 0.5, 1.0, 1.0]
plt.scatter(range(4), fastvar_ratios, color='red', marker='X', s=100, label='FastVAR (Static)')

# 5. 美化圖表
plt.title('Learned Pruning Policy: Distribution of Actions ($p_k$)', fontsize=14)
plt.ylabel('Pruning Ratio ($p_k$)', fontsize=12)
plt.xlabel('Decision Steps ($k$)', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig("violinplot.png")


import pandas as pd
import matplotlib.pyplot as plt

# 1. Define category order and mapping logic in English
category_order = [
    'Full Pruning', 
    'Pruning 0.999 ~ 0.75', 
    'Pruning 0.75 ~ 0.5', 
    'Pruning 0.5 ~ 0.25', 
    'Pruning 0.25 ~ 0.001', 
    'Full Retention'
]

def categorize_p(val):
    if val >= 1.0: 
        return 'Full Pruning'
    elif 0.75 <= val < 1.0: 
        return 'Pruning 0.999 ~ 0.75'
    elif 0.5 <= val < 0.75: 
        return 'Pruning 0.75 ~ 0.5'
    elif 0.25 <= val < 0.5: 
        return 'Pruning 0.5 ~ 0.25'
    elif 0.001 <= val < 0.25: 
        return 'Pruning 0.25 ~ 0.001'
    else: 
        return 'Full Retention'

# 2. Data Processing
rows = []
scales = ['K-3', 'K-2', 'K-1', 'K']

# Assuming 'data' is your dictionary from JSON
for p_values in data.values():
    for i, p_val in enumerate(p_values):
        if i < len(scales):
            rows.append({'Scale': scales[i], 'Category': categorize_p(p_val)})

df = pd.DataFrame(rows)

# 3. Calculate Percentages
dist_df = df.groupby(['Scale', 'Category']).size().unstack(fill_value=0)
dist_df = dist_df.div(dist_df.sum(axis=1), axis=0) * 100

# 4. Reindex to ensure the specific order and scale sequence
dist_df = dist_df.reindex(columns=category_order, fill_value=0)
dist_df = dist_df.reindex(['K-3', 'K-2', 'K-1', 'K']) 

# 5. Plotting
# Color Palette: From Red (High Pruning) to Green (Low Pruning/Retention)
colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#66bd63']

ax = dist_df.plot(kind='bar', stacked=True, figsize=(11, 7), 
                  color=colors, edgecolor='white', width=0.7)

# Title and Labels
plt.title('Pruning Strategy Frequency per Scale', fontsize=14, pad=15)
plt.ylabel('Percentage of Samples (%)', fontsize=12)
plt.xlabel('Decision Steps ($k$)', fontsize=12)

# Legend Configuration
plt.legend(title='Policy Action', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("bar.png")
"""
╔══════════════════════════════════════════════════════════════════╗
║     SAMATRIX RESUMEFORGE 2026 — Complete EDA & Preprocessing     ║
║     Step 1: Exploratory Data Analysis                            ║
╚══════════════════════════════════════════════════════════════════╝

Run this script standalone to generate all EDA visualizations.
All plots are saved in the 'outputs/eda/' folder.
"""

import sys, io, os, re, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ── Must set CWD to script directory so config resolves correctly ──
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH, EDA_DIR, PLOTS_DIR

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ── Setup ─────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

PALETTE = [
    "#6C63FF","#FF6584","#43C6AC","#F7971E","#FFD200",
    "#7F7FD5","#86A8E7","#91EAE4","#FC5C7D","#6A3093",
    "#1CB5E0","#000851","#0f0c29","#302b63","#24243e",
    "#f12711","#f5af19","#11998e","#38ef7d","#00c6ff",
    "#0072ff","#ee0979","#ff6a00","#833ab4","#fd1d1d",
]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.facecolor': '#0f0f1a',
    'figure.facecolor': '#0f0f1a',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'axes.edgecolor': '#333355',
    'grid.color': '#1a1a2e',
    'grid.alpha': 0.5,
})

# DATA_PATH is imported from config

print("=" * 65)
print("  SAMATRIX RESUMEFORGE 2026 — EDA Module")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Columns: {df.columns.tolist()}")
print(f"    Categories: {df['Category'].nunique()}")
print(f"    Missing values:\n{df.isnull().sum()}")

# Duplicates
print(f"\n    Duplicate rows: {df.duplicated().sum()}")
print(f"    Duplicate resume text: {df['Resume_str'].duplicated().sum()}")

# Text length features
df['text_length'] = df['Resume_str'].fillna('').str.len()
df['word_count'] = df['Resume_str'].fillna('').str.split().str.len()
df['char_per_word'] = (df['text_length'] / df['word_count'].clip(lower=1)).fillna(0).replace([np.inf, -np.inf], 0)

print(f"\n    Text length stats:\n{df['text_length'].describe()}")
print(f"\n    Word count stats:\n{df['word_count'].describe()}")

# ══════════════════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
print("\n[2] Plotting class distribution...")
cat_counts = df['Category'].value_counts().reset_index()
cat_counts.columns = ['Category', 'Count']

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor('#0f0f1a')

# Bar chart
colors_bar = [PALETTE[i % len(PALETTE)] for i in range(len(cat_counts))]
bars = axes[0].barh(cat_counts['Category'], cat_counts['Count'],
                    color=colors_bar, edgecolor='none', height=0.7)
axes[0].set_xlabel('Number of Resumes', fontsize=12, labelpad=10)
axes[0].set_title('Resume Count per Category', fontsize=14, pad=15, fontweight='bold', color='#e0e0e0')
axes[0].invert_yaxis()
axes[0].set_facecolor('#0f0f1a')
for bar, count in zip(bars, cat_counts['Count']):
    axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{count}', va='center', fontsize=9, color='#a0a0c0')

# Donut chart
wedges, texts, autotexts = axes[1].pie(
    cat_counts['Count'], labels=None,
    colors=colors_bar, autopct='%1.1f%%',
    startangle=140, pctdistance=0.85,
    wedgeprops=dict(width=0.5, edgecolor='#0f0f1a', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(7)
    at.set_color('#e0e0e0')
axes[1].set_title('Class Distribution (Donut)', fontsize=14, pad=15, fontweight='bold', color='#e0e0e0')
legend = axes[1].legend(wedges, cat_counts['Category'],
                         loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                         fontsize=8, framealpha=0.2, labelcolor='#e0e0e0',
                         facecolor='#1a1a2e')

plt.suptitle('SAMATRIX ResumeForge 2026 — Class Distribution', fontsize=16,
             fontweight='bold', color='#6C63FF', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 01_class_distribution.png")

# ══════════════════════════════════════════════════════════════════
# 3. TEXT LENGTH DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════
print("\n[3] Plotting text length distributions...")
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
fig.patch.set_facecolor('#0f0f1a')

for ax, col, title, color in zip(
    axes,
    ['text_length', 'word_count', 'char_per_word'],
    ['Character Count', 'Word Count', 'Avg Chars per Word'],
    ['#6C63FF', '#FF6584', '#43C6AC']
):
    ax.hist(df[col], bins=50, color=color, alpha=0.85, edgecolor='none')
    ax.axvline(df[col].mean(), color='#FFD200', linewidth=2, linestyle='--',
               label=f'Mean: {df[col].mean():.0f}')
    ax.axvline(df[col].median(), color='#FF6584', linewidth=2, linestyle=':',
               label=f'Median: {df[col].median():.0f}')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#e0e0e0')
    ax.legend(fontsize=9, framealpha=0.2, labelcolor='#e0e0e0', facecolor='#1a1a2e')
    ax.set_facecolor('#0f0f1a')

plt.suptitle('Text Length Analysis', fontsize=15, fontweight='bold', color='#6C63FF')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '02_text_length_distribution.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 02_text_length_distribution.png")

# Text length per category (boxplot)
fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

category_order = df.groupby('Category')['word_count'].median().sort_values(ascending=False).index
data_by_cat = [df[df['Category'] == cat]['word_count'].values for cat in category_order]
bp = ax.boxplot(data_by_cat, labels=category_order, patch_artist=True,
                notch=True, showfliers=True,
                flierprops=dict(marker='o', color='#FF6584', alpha=0.4, markersize=3))

for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(PALETTE[i % len(PALETTE)])
    patch.set_alpha(0.8)
for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='#e0e0e0', linewidth=1.2)
plt.setp(bp['medians'], color='#FFD200', linewidth=2)

ax.set_xlabel('Category', fontsize=12, labelpad=10)
ax.set_ylabel('Word Count', fontsize=12, labelpad=10)
ax.set_title('Word Count Distribution per Resume Category', fontsize=14, fontweight='bold', color='#e0e0e0')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '03_wordcount_per_category.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 03_wordcount_per_category.png")

# ══════════════════════════════════════════════════════════════════
# 4. WORD FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n[4] Computing word frequencies...")
stop_words = set(stopwords.words('english'))
extra_stops = {'experience', 'work', 'year', 'years', 'one', 'two', 'three',
               'also', 'etc', 'including', 'name', 'date', 'address',
               'email', 'phone', 'resume', 'cv', 'page'}
stop_words.update(extra_stops)

def get_top_words(texts, n=20, remove_stops=True):
    all_words = []
    for text in texts:
        text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
        words = text.split()
        if remove_stops:
            words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

top_words = get_top_words(df['Resume_str'])
words, counts = zip(*top_words)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')
colors = [PALETTE[i % len(PALETTE)] for i in range(len(words))]
bars = ax.barh(list(words), list(counts), color=colors, edgecolor='none')
ax.invert_yaxis()
ax.set_title('Top 20 Most Frequent Words (All Categories)', fontsize=14,
             fontweight='bold', color='#e0e0e0')
ax.set_xlabel('Frequency', fontsize=12)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
            f'{count:,}', va='center', fontsize=9, color='#a0a0c0')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '04_top_words_overall.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 04_top_words_overall.png")

# ══════════════════════════════════════════════════════════════════
# 5. WORDCLOUD
# ══════════════════════════════════════════════════════════════════
print("\n[5] Generating WordClouds...")

def make_wordcloud(text, title, filename, bg='#0f0f1a', colormap='plasma'):
    wc = WordCloud(
        width=1200, height=600, background_color=bg,
        colormap=colormap, max_words=200,
        stopwords=stop_words, collocations=False,
        random_state=42
    ).generate(text)
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(bg)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', color='#e0e0e0', pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor=bg)
    plt.close()

all_text = ' '.join(df['Resume_str'].fillna(''))
make_wordcloud(all_text, 'WordCloud — All Resumes', os.path.join(EDA_DIR, '05_wordcloud_all.png'))
print("    OK: WordCloud (all) saved")

selected_cats = ['INFORMATION-TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'ENGINEERING',
                 'HR', 'SALES', 'TEACHER', 'DESIGNER']
colormaps = ['Blues', 'Greens', 'YlOrBr', 'copper', 'RdPu', 'OrRd', 'PuBu', 'cool']

fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.patch.set_facecolor('#0f0f1a')
for ax, cat, cmap in zip(axes.flat, selected_cats, colormaps):
    cat_text = ' '.join(df[df['Category'] == cat]['Resume_str'].fillna(''))
    if cat_text.strip():
        wc = WordCloud(width=600, height=300, background_color='#0f0f1a',
                      colormap=cmap, max_words=100,
                      stopwords=stop_words, random_state=42).generate(cat_text)
        ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(cat.replace('-', '\n'), fontsize=10, fontweight='bold', color='#e0e0e0', pad=5)

plt.suptitle('Category-Specific WordClouds', fontsize=18, fontweight='bold', color='#6C63FF')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '06_wordcloud_per_category.png'), dpi=120, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: WordCloud (per category) saved")

# ══════════════════════════════════════════════════════════════════
# 6. N-GRAM ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n[6] Computing bigrams/trigrams...")
from sklearn.feature_extraction.text import CountVectorizer

def get_top_ngrams(corpus, n_gram_range, top_n=15):
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words='english',
                          max_features=5000).fit(corpus)
    bag = vec.transform(corpus)
    freq = pd.DataFrame({'ngram': vec.get_feature_names_out(),
                         'count': bag.toarray().sum(axis=0)})
    return freq.nlargest(top_n, 'count')

bigrams = get_top_ngrams(df['Resume_str'].fillna(''), (2, 2))
trigrams = get_top_ngrams(df['Resume_str'].fillna(''), (3, 3))

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
fig.patch.set_facecolor('#0f0f1a')

for ax, df_ng, title, color in zip(
    axes,
    [bigrams, trigrams],
    ['Top 15 Bigrams', 'Top 15 Trigrams'],
    ['#6C63FF', '#FF6584']
):
    ax.set_facecolor('#0f0f1a')
    bars = ax.barh(df_ng['ngram'], df_ng['count'], color=color, alpha=0.85, edgecolor='none')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight='bold', color='#e0e0e0')
    ax.set_xlabel('Count', fontsize=11)
    for bar, count in zip(bars, df_ng['count']):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{count:,}', va='center', fontsize=9, color='#a0a0c0')

plt.suptitle('N-Gram Analysis — All Resumes', fontsize=15, fontweight='bold', color='#6C63FF')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '07_ngram_analysis.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 07_ngram_analysis.png")

# ══════════════════════════════════════════════════════════════════
# 7. TF-IDF HEATMAP
# ══════════════════════════════════════════════════════════════════
print("\n[7] Building TF-IDF category heatmap...")
from sklearn.feature_extraction.text import TfidfVectorizer

selected_cats_hm = ['INFORMATION-TECHNOLOGY', 'HEALTHCARE', 'FINANCE', 'HR',
                    'SALES', 'TEACHER', 'ENGINEERING', 'CHEF']
df_hm = df[df['Category'].isin(selected_cats_hm)]

tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df_hm['Resume_str'].fillna(''))
feat_names = tfidf.get_feature_names_out()

tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feat_names)
tfidf_df['Category'] = df_hm['Category'].values
cat_means = tfidf_df.groupby('Category').mean()

top_terms = []
for cat in cat_means.index:
    top5 = cat_means.loc[cat].nlargest(5).index.tolist()
    top_terms.extend(top5)
top_terms = list(dict.fromkeys(top_terms))

heatmap_data = cat_means[top_terms]

fig, ax = plt.subplots(figsize=(28, 8))
fig.patch.set_facecolor('#0f0f1a')

sns.heatmap(heatmap_data, ax=ax, cmap='viridis', annot=False,
            linewidths=0.5, linecolor='#0f0f1a',
            cbar_kws={'label': 'Mean TF-IDF Score'})
ax.set_title('TF-IDF Term Importance Heatmap by Category', fontsize=14,
             fontweight='bold', color='#e0e0e0', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '08_tfidf_heatmap.png'), dpi=120, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 08_tfidf_heatmap.png")

# ══════════════════════════════════════════════════════════════════
# 8. NOISE ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n[8] Noise analysis...")
df['has_html'] = df['Resume_str'].str.contains(r'<[^>]+>', regex=True)
df['has_url'] = df['Resume_str'].str.contains(r'http[s]?://', regex=True)
df['has_email'] = df['Resume_str'].str.contains(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b', regex=True)
df['is_very_short'] = df['word_count'] < 50

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.patch.set_facecolor('#0f0f1a')
for ax, col, title, color in zip(
    axes,
    ['has_html', 'has_url', 'has_email', 'is_very_short'],
    ['HTML Tags', 'URLs', 'Email Addrs', 'Very Short (<50w)'],
    ['#6C63FF', '#FF6584', '#43C6AC', '#F7971E']
):
    vals = [df[col].sum(), len(df) - df[col].sum()]
    ax.pie(vals, labels=['Yes', 'No'], colors=[color, '#2a2a4a'],
           autopct='%1.1f%%', startangle=90,
           wedgeprops=dict(edgecolor='#0f0f1a', linewidth=2))
    ax.set_title(title, fontsize=11, fontweight='bold', color='#e0e0e0', pad=8)

plt.suptitle('Data Quality / Noise Analysis', fontsize=15, fontweight='bold', color='#6C63FF')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, '09_noise_analysis.png'), dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
plt.close()
print("    OK: 09_noise_analysis.png")

print("\n" + "=" * 65)
print("  EDA COMPLETE -- All plots saved in outputs/eda/")
print("=" * 65)

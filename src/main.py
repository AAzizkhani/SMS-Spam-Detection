# NLP Preprocessing Pipeline | SMS Spam Dataset

import os
import sys
import re
import subprocess

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')

RAW_DATA_PATH = os.path.join(RAW_DIR, 'sms.tsv')
PREPROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, 'sms_spam_preprocessed.csv')
ANALYSIS_PNG_PATH = os.path.join(FIGURES_DIR, 'nlp_analysis.png')
EMBEDDING_SCRIPT_PATH = os.path.join(BASE_DIR, 'embedding.py')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

if not os.path.exists(RAW_DATA_PATH):
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df.to_csv(RAW_DATA_PATH, sep='\t', index=False, header=False)
else:
    df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=['label', 'message'])

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"\nLabel Distribution:\n{df['label'].value_counts()}")
print(f"\nSample Data:\n{df.head()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")


def clean_text(text):
    """Basic text cleaning"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text):
    """Tokenization"""
    return word_tokenize(text)


def remove_stopwords(tokens):
    """Remove stopwords"""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


print("\nApplying Preprocessing:")

df['cleaned_text'] = df['message'].apply(clean_text)
df['tokens'] = df['cleaned_text'].apply(tokenize_text)
df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)

print("Cleaning and Tokenization done!")


porter = PorterStemmer()


def apply_stemming(tokens):
    """Apply Porter Stemmer"""
    return [porter.stem(token) for token in tokens]


df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(apply_stemming)
df['stemmed_text'] = df['stemmed_tokens'].apply(lambda x: ' '.join(x))

print("Stemming done!")


lemmatizer = WordNetLemmatizer()


def apply_lemmatization(tokens):
    """Apply WordNet Lemmatizer"""
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]


df['lemmatized_tokens'] = df['tokens_no_stopwords'].apply(apply_lemmatization)
df['lemmatized_text'] = df['lemmatized_tokens'].apply(lambda x: ' '.join(x))

print("Lemmatization done!")


print("\n" + "=" * 60)
print("COMPARISON: Stemming vs Lemmatization")
print("=" * 60)

comparison_words = [
    ['running', 'runs', 'ran'],
    ['better', 'good', 'best'],
    ['studies', 'studying', 'studied'],
    ['flies', 'flying', 'flew'],
    ['caring', 'cares', 'cared']
]

print(f"\n{'Word':<15} {'Stemmed':<20} {'Lemmatized':<20}")
print("-" * 55)

for word_group in comparison_words:
    for word in word_group:
        stemmed = porter.stem(word)
        lemmatized = lemmatizer.lemmatize(word, pos='v')
        print(f"{word:<15} {stemmed:<20} {lemmatized:<20}")
    print()


print("\nVocabulary Size Comparison:")

original_vocab = set(word for tokens in df['tokens_no_stopwords'] for word in tokens)
stemmed_vocab = set(word for tokens in df['stemmed_tokens'] for word in tokens)
lemmatized_vocab = set(word for tokens in df['lemmatized_tokens'] for word in tokens)

print(f"Original Vocabulary Size:    {len(original_vocab)}")
print(f"Stemmed Vocabulary Size:     {len(stemmed_vocab)}")
print(f"Lemmatized Vocabulary Size:  {len(lemmatized_vocab)}")
print(f"\nReduction by Stemming:       {((len(original_vocab) - len(stemmed_vocab)) / len(original_vocab) * 100):.1f}%")
print(f"Reduction by Lemmatization:  {((len(original_vocab) - len(lemmatized_vocab)) / len(original_vocab) * 100):.1f}%")

df['avg_stem_len'] = df['stemmed_tokens'].apply(
    lambda x: np.mean([len(w) for w in x]) if x else 0
)
df['avg_lemma_len'] = df['lemmatized_tokens'].apply(
    lambda x: np.mean([len(w) for w in x]) if x else 0
)

print(f"\nAverage token length (Stemmed):     {df['avg_stem_len'].mean():.2f}")
print(f"Average token length (Lemmatized):  {df['avg_lemma_len'].mean():.2f}")


print("\nSample Messages Comparison:")
for i in range(3):
    print(f"\n--- Sample {i + 1} [{df['label'].iloc[i]}] ---")
    print(f"Original:     {df['message'].iloc[i]}")
    print(f"Cleaned:      {df['cleaned_text'].iloc[i]}")
    print(f"Tokens:       {df['tokens_no_stopwords'].iloc[i]}")
    print(f"Stemmed:      {df['stemmed_tokens'].iloc[i]}")
    print(f"Lemmatized:   {df['lemmatized_tokens'].iloc[i]}")


sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NLP Preprocessing Analysis - SMS Spam Dataset', fontsize=14, fontweight='bold')

label_counts = df['label'].value_counts()
axes[0, 0].pie(
    label_counts,
    labels=label_counts.index,
    autopct='%1.1f%%',
    colors=['#2ecc71', '#e74c3c']
)
axes[0, 0].set_title('Label Distribution')

df['msg_length'] = df['message'].apply(len)
axes[0, 1].hist(
    [df[df['label'] == 'ham']['msg_length'],
     df[df['label'] == 'spam']['msg_length']],
    bins=30,
    label=['Ham', 'Spam'],
    color=['#2ecc71', '#e74c3c'],
    alpha=0.7
)
axes[0, 1].set_title('Message Length Distribution')
axes[0, 1].set_xlabel('Length')
axes[0, 1].legend()

vocab_data = {
    'Original': len(original_vocab),
    'Stemmed': len(stemmed_vocab),
    'Lemmatized': len(lemmatized_vocab)
}
axes[1, 0].bar(
    vocab_data.keys(),
    vocab_data.values(),
    color=['#3498db', '#e67e22', '#9b59b6']
)
axes[1, 0].set_title('Vocabulary Size Comparison')
axes[1, 0].set_ylabel('Unique Words')

for i, (k, v) in enumerate(vocab_data.items()):
    axes[1, 0].text(i, v + 50, str(v), ha='center', fontweight='bold')

df['token_count'] = df['tokens_no_stopwords'].apply(len)
df['stem_count'] = df['stemmed_tokens'].apply(len)
df['lemma_count'] = df['lemmatized_tokens'].apply(len)

axes[1, 1].boxplot(
    [df['token_count'], df['stem_count'], df['lemma_count']],
    labels=['Original\nTokens', 'Stemmed', 'Lemmatized']
)
axes[1, 1].set_title('Token Count Distribution')
axes[1, 1].set_ylabel('Number of Tokens')

plt.tight_layout()
plt.savefig(ANALYSIS_PNG_PATH, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as '{ANALYSIS_PNG_PATH}'")


df_final = df[
    ['label', 'message', 'cleaned_text',
     'tokens_no_stopwords', 'stemmed_text', 'lemmatized_text']
]

df_final.to_csv(PREPROCESSED_CSV_PATH, index=False)

print(f"\nFinal dataset saved as '{PREPROCESSED_CSV_PATH}'")
print(f"\nFinal Dataset Shape: {df_final.shape}")
print(f"\nColumns: {list(df_final.columns)}")


print("\nRunning embedding.py ...")
subprocess.run([sys.executable, EMBEDDING_SCRIPT_PATH], check=True)

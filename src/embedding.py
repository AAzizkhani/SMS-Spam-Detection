# BOW, TF-IDF, Word2Vec on SMS Spam Collection

import os
import ast
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sms_spam_preprocessed.csv')

EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'embeddings')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')

BOW_PATH = os.path.join(EMBEDDINGS_DIR, 'bow_features.csv')
TFIDF_PATH = os.path.join(EMBEDDINGS_DIR, 'tfidf_features.csv')
W2V_DOC_PATH = os.path.join(EMBEDDINGS_DIR, 'word2vec_doc_vectors.csv')
SUMMARY_PATH = os.path.join(EMBEDDINGS_DIR, 'embedding_summary.csv')
W2V_MODEL_PATH = os.path.join(MODELS_DIR, 'word2vec.model')

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found. Run src/main.py first."
    )

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {df.shape[0]} rows | columns: {list(df.columns)}")

texts = df['lemmatized_text'].fillna('').astype(str)
labels = df['label']


def parse_tokens(cell):
    try:
        return ast.literal_eval(cell)
    except Exception:
        return str(cell).split()


token_lists = df['tokens_no_stopwords'].apply(parse_tokens)


print("\n[1/3] BOW ...")

bow_vectorizer = CountVectorizer(max_features=5000, min_df=2)
bow_matrix = bow_vectorizer.fit_transform(texts)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)
bow_df.insert(0, 'label', labels.values)
bow_df.to_csv(BOW_PATH, index=False)

print(f"  Shape: {bow_df.shape} -> saved {BOW_PATH}")


print("\n[2/3] TF-IDF ...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    sublinear_tf=True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)
tfidf_df.insert(0, 'label', labels.values)
tfidf_df.to_csv(TFIDF_PATH, index=False)

print(f"  Shape: {tfidf_df.shape} -> saved {TFIDF_PATH}")


print("\n[3/3] Word2Vec ...")

w2v_model = Word2Vec(
    sentences=token_lists.tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=10,
    sg=1
)

w2v_model.save(W2V_MODEL_PATH)
print(f"  Vocabulary size: {len(w2v_model.wv)} words -> saved {W2V_MODEL_PATH}")


def doc_vector(tokens):
    vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)


doc_vectors = np.vstack(token_lists.apply(doc_vector).values)
cols = [f'w2v_{i}' for i in range(100)]

w2v_df = pd.DataFrame(doc_vectors, columns=cols)
w2v_df.insert(0, 'label', labels.values)
w2v_df.to_csv(W2V_DOC_PATH, index=False)

print(f"  Doc-vector shape: {w2v_df.shape} -> saved {W2V_DOC_PATH}")


summary = pd.DataFrame({
    'method': ['BOW', 'TF-IDF', 'Word2Vec'],
    'n_docs': [bow_df.shape[0], tfidf_df.shape[0], w2v_df.shape[0]],
    'n_features': [bow_df.shape[1] - 1, tfidf_df.shape[1] - 1, w2v_df.shape[1] - 1],
    'sparse': [True, True, False],
    'semantic': [False, False, True],
    'output_file': [
        'outputs/embeddings/bow_features.csv',
        'outputs/embeddings/tfidf_features.csv',
        'outputs/embeddings/word2vec_doc_vectors.csv'
    ],
})

summary.to_csv(SUMMARY_PATH, index=False)

print("\n── Summary ──────────────────────────────────")
print(summary.to_string(index=False))
print("\nAll done.")

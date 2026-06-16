# SMS Spam Detection | NLP Preprocessing & Embedding Pipeline

A complete NLP pipeline for the **SMS Spam Collection** dataset, including:

- text preprocessing
- stemming and lemmatization
- exploratory visualization
- classical vectorization with **Bag of Words (BoW)** and **TF-IDF**
- distributed representation using **Word2Vec**

The project starts from raw SMS messages, cleans and normalizes the text, saves a processed dataset, and then generates multiple embedding formats ready for machine learning tasks such as **spam detection**.

---

## Dataset

- **Dataset:** SMS Spam Collection
- **Source URL used in code:**  
  `https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv`
- **Columns:**
  - `label` → class of the message (`ham` or `spam`)
  - `message` → original SMS text

---

## Project Goals

This project is designed to build a practical NLP workflow for spam detection by:

1. loading the raw SMS dataset
2. cleaning noisy text
3. tokenizing messages
4. removing stopwords
5. applying stemming and lemmatization
6. analyzing vocabulary changes
7. visualizing preprocessing statistics
8. converting text into machine-readable vectors using:
   - **BoW**
   - **TF-IDF**
   - **Word2Vec**

---

## Features

### Preprocessing
Implemented in `main.py`:

- lowercase conversion
- URL removal
- email removal
- number removal
- punctuation/special character removal
- whitespace normalization
- tokenization with NLTK
- stopword removal
- stemming with `PorterStemmer`
- lemmatization with `WordNetLemmatizer`

### Analysis
The preprocessing script also performs:

- dataset shape and info display
- label distribution check
- missing value analysis
- stemming vs lemmatization comparison
- vocabulary size comparison
- average token length analysis
- sample message comparison

### Visualization
The project generates a figure named:

- `nlp_analysis.png`

It includes:
- label distribution
- message length distribution
- vocabulary size comparison
- token count distribution

### Embedding / Vectorization
Implemented in `embedding.py`:

- **Bag of Words (BoW)** using `CountVectorizer`
- **TF-IDF** using `TfidfVectorizer`
- **Word2Vec** using `gensim.models.Word2Vec`
- document-level Word2Vec vectors using **mean pooling**

---

## Project Structure

```bash
SMS-Spam-Detection/
│
├── data/
│   ├── raw/
│   │   └── sms.tsv                         # Raw dataset
│   └── processed/
│       └── sms_spam_preprocessed.csv       # Preprocessed dataset
│
├── outputs/
│   ├── figures/
│   │   └── nlp_analysis.png                # NLP analysis plots
│   ├── embeddings/
│   │   ├── bow_features.csv                # Bag of Words features
│   │   ├── tfidf_features.csv              # TF-IDF features
│   │   ├── word2vec_doc_vectors.csv        # Document-level Word2Vec vectors
│   │   └── embedding_summary.csv           # Embedding methods summary
│   └── models/
│       └── word2vec.model                  # Trained Word2Vec model
│
├── src/
│   ├── main.py                             # Text preprocessing + visualization pipeline
│   └── embedding.py                        # BoW, TF-IDF, and Word2Vec generation
│
├── README.md
└── requirements.txt

```

---

## How the Pipeline Works

### Step 1 — Preprocessing (`main.py`)
The script:

- downloads the SMS dataset from a URL
- loads it into a pandas DataFrame
- cleans the text
- tokenizes the messages
- removes stopwords
- applies stemming
- applies lemmatization
- compares stemming and lemmatization
- creates visualizations
- saves the processed dataset as:

```bash
sms_spam_preprocessed.csv
```

At the end of the script, it automatically runs:

```bash
python embedding.py
```

So running `main.py` is enough to execute the full pipeline.

---

### Step 2 — Embedding (`embedding.py`)
This script reads:

```bash
sms_spam_preprocessed.csv
```

Then it creates:

1. **BoW features** → `bow_features.csv`
2. **TF-IDF features** → `tfidf_features.csv`
3. **Word2Vec model** → `word2vec.model`
4. **Word2Vec document vectors** → `word2vec_doc_vectors.csv`
5. **Embedding summary** → `embedding_summary.csv`

---

## Requirements

- Python 3.8+
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- scikit-learn
- gensim

---

## Installation

Install dependencies:

```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn gensim
```

If needed, you can also install from a `requirements.txt` file.

---

## NLTK Resources

The script downloads the required NLTK resources automatically:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
```

---

## Usage

Run the full pipeline with:

```bash
python main.py
```

This will:

- preprocess the SMS dataset
- generate analysis plots
- save the cleaned dataset
- automatically run `embedding.py`
- generate all embedding outputs

If you already have the preprocessed file and only want embeddings:

```bash
python embedding.py
```

---

## Output Files

### 1. Preprocessed Dataset
**File:** `sms_spam_preprocessed.csv`

Contains these columns:

| Column | Description |
|--------|-------------|
| `label` | Message label (`ham` or `spam`) |
| `message` | Original message |
| `cleaned_text` | Cleaned version of the message |
| `tokens_no_stopwords` | Token list after stopword removal |
| `stemmed_text` | Stemmed text |
| `lemmatized_text` | Lemmatized text |

---

### 2. BoW Features
**File:** `bow_features.csv`

- generated with `CountVectorizer`
- uses:
  - `max_features=5000`
  - `min_df=2`

This file contains one row per document and one column per vocabulary term, plus the `label` column.

---

### 3. TF-IDF Features
**File:** `tfidf_features.csv`

- generated with `TfidfVectorizer`
- uses:
  - `max_features=5000`
  - `min_df=2`
  - `sublinear_tf=True`

This file contains weighted term features for each message, plus the `label` column.

---

### 4. Word2Vec Model
**File:** `word2vec.model`

Word2Vec is trained with:

- `vector_size=100`
- `window=5`
- `min_count=2`
- `workers=4`
- `epochs=10`
- `sg=1` → Skip-gram architecture

---

### 5. Word2Vec Document Vectors
**File:** `word2vec_doc_vectors.csv`

- Each message is represented by a **100-dimensional dense vector**
- Document vectors are created using **mean pooling** over token embeddings
- Includes the `label` column

---

### 6. Embedding Summary
**File:** `embedding_summary.csv`

Contains a summary table for:

- method name
- number of documents
- number of features
- sparse/dense representation
- semantic capability
- output file name

---

## Preprocessing Details

### Text Cleaning Rules
The `clean_text()` function applies the following transformations:

- convert text to lowercase
- remove URLs
- remove email addresses
- remove digits
- keep only English letters and whitespace
- remove extra spaces

Example transformations:

- `"WIN a FREE ticket!!!"` → `"win a free ticket"`
- `"Visit http://abc.com now"` → `"visit now"`

---

## Stemming vs Lemmatization

The project compares both normalization methods:

### Stemming
Uses:
- `PorterStemmer`

Example:
- `studies` → `studi`
- `running` → `run`

### Lemmatization
Uses:
- `WordNetLemmatizer` with `pos='v'`

Example:
- `studies` → `study`
- `running` → `run`

In general:

- **Stemming** is faster and more aggressive
- **Lemmatization** is more linguistically meaningful

---

## Visualization

The generated file:

```bash
nlp_analysis.png
```

contains four plots:

1. **Label Distribution**
2. **Message Length Distribution**
3. **Vocabulary Size Comparison**
4. **Token Count Distribution**

This helps analyze:
- class imbalance
- message length patterns
- vocabulary reduction after normalization
- token distribution changes

---

## Embedding Methods Comparison

| Method | Sparse | Dense | Semantic Information | Output |
|--------|--------|-------|----------------------|--------|
| BoW | Yes | No | No | `bow_features.csv` |
| TF-IDF | Yes | No | Limited | `tfidf_features.csv` |
| Word2Vec | No | Yes | Yes | `word2vec_doc_vectors.csv` |

### Notes
- **BoW** counts token frequencies
- **TF-IDF** downweights common words and emphasizes informative ones
- **Word2Vec** captures semantic relationships between words

---

## Error Handling

`embedding.py` checks whether the preprocessing output exists first:

```python
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"'{CSV_PATH}' not found. Run main.py first."
    )
```

So if `sms_spam_preprocessed.csv` is missing, you must run:

```bash
python src/main.py
```

before running embeddings.

---

## Example Workflow

```text
Raw SMS Messages
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stopword Removal
      ↓
Stemming / Lemmatization
      ↓
Save Preprocessed Dataset
      ↓
BoW / TF-IDF / Word2Vec
      ↓
Machine Learning Ready Features
```

---

## Possible Extensions

You can improve this project further by adding:

- train/test split
- model training for spam classification
- evaluation metrics:
  - accuracy
  - precision
  - recall
  - F1-score
- confusion matrix
- logistic regression / SVM / Naive Bayes classifiers
- hyperparameter tuning
- WordCloud visualization
- FastText / GloVe embeddings
- transformer-based embeddings such as BERT

---

## Example `.gitignore`

```gitignore
__pycache__/
*.pyc
.venv/
env/
.idea/
.vscode/
.DS_Store

# generated outputs
sms_spam_preprocessed.csv
bow_features.csv
tfidf_features.csv
word2vec_doc_vectors.csv
embedding_summary.csv
word2vec.model
nlp_analysis.png
```

---

## Author

**Ali Azizkhani**

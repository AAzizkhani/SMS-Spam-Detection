# SMS Spam Detection - NLP Pipeline

A complete NLP preprocessing pipeline for the UCI SMS Spam Collection dataset.

## Dataset
- **Source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Size:** 5,574 SMS messages
- **Classes:** Ham (~86.6%) | Spam (~13.4%)

## Features
- Text cleaning (URLs, emails, numbers, punctuation removal)
- Tokenization
- Stopword removal
- Stemming (PorterStemmer)
- Lemmatization (WordNetLemmatizer)
- Visualization of key statistics

## Project Structure
project/
│
├── data/
│   └── sms_spam_preprocessed.csv  # Preprocessed output
│
├── main.py                        # Main pipeline script
├── nlp_analysis.png               # Generated visualizations
└── README.md

## Requirements
Python 3.8+
nltk
pandas
matplotlib

## Installation
bash
pip install nltk pandas matplotlib

Then download required NLTK packages:
python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

## Usage
bash
python main.py

## Output
The preprocessed CSV contains these columns:

|         Column        |          Description         |
|-----------------------|------------------------------|
| `label`               | Ham or Spam                  |
| `message`             | Original message             |
| `cleaned_text`        | After cleaning               |
| `tokens_no_stopwords` | Tokenized, stopwords removed |
| `stemmed_text`        | After Porter Stemming        |
| `lemmatized_text`     | After WordNet Lemmatization  |

## Results

| Method        | Vocabulary Reduction |
|---------------|----------------------|
| Stemming      |          ~27%        |
| Lemmatization |          ~16%        |

Stemming is more aggressive but may produce non-meaningful tokens (e.g. `studi`).  
Lemmatization preserves meaning (e.g. `study`).

## Visualization
`nlp_analysis.png` includes:
- Label distribution
- Message length histogram
- Vocabulary size comparison
- Token count boxplot

# By Ali Azizkhani

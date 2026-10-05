"""
SAMATRIX RESUMEFORGE 2026
Step 2: Preprocessing Pipeline (Reproducible)
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Technical tokens to preserve (must NOT be removed)
TECH_TOKENS = {
    'python', 'java', 'sql', 'aws', 'c++', 'javascript', 'typescript',
    'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'sklearn',
    'nlp', 'ml', 'ai', 'tableau', 'powerbi', 'excel', 'sap', 'erp',
    'html', 'css', 'react', 'angular', 'nodejs', 'docker', 'kubernetes',
    'git', 'linux', 'agile', 'scrum', 'devops', 'ci', 'cd', 'api',
    'gcp', 'azure', 'hadoop', 'spark', 'kafka', 'mongodb', 'postgresql',
    'mysql', 'redis', 'elasticsearch', 'r', 'matlab', 'scala', 'swift',
    'kotlin', 'ruby', 'php', 'cpp', 'net', 'blockchain', 'cybersecurity',
    'iot', 'ros', 'cad', 'autocad', 'solidworks', 'revit', 'photoshop',
    'illustrator', 'figma', 'sklearn', 'pca', 'svm', 'lstm', 'gru',
    'bert', 'gpt', 'transformer', 'cnn', 'rnn', 'xgboost', 'lightgbm',
    'mba', 'bca', 'btech', 'mtech', 'phd', 'bsc', 'msc',
}


def clean_resume(text: str, lemmatize: bool = True) -> str:
    """
    Reproducible resume text cleaning pipeline.
    Preserves technical tokens, removes noise.
    """
    if not isinstance(text, str):
        return ''

    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Replace URLs with token
    text = re.sub(r'https?://\S+|www\.\S+', ' url_token ', text)

    # 3. Replace emails with token
    text = re.sub(r'\b[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', ' email_token ', text)

    # 4. Replace phone numbers with token
    text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{8,}', ' phone_token ', text)

    # 5. Normalize C++ and .NET before lowercasing
    text = text.replace('C++', 'cpp').replace('c++', 'cpp')
    text = text.replace('.NET', 'dotnet').replace('.net', 'dotnet')
    text = text.replace('Node.js', 'nodejs').replace('node.js', 'nodejs')

    # 6. Lowercase
    text = text.lower()

    # 7. Remove encoding artifacts / non-ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 8. Remove special characters but keep alphanumerics + some punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Tokenize and apply stopword removal + lemmatization
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if len(token) < 2:
            continue
        # Preserve tech tokens
        if token in TECH_TOKENS:
            cleaned_tokens.append(token)
            continue
        # Remove stopwords
        if token in STOP_WORDS:
            continue
        # Lemmatize
        if lemmatize:
            token = lemmatizer.lemmatize(token)
        if len(token) >= 2:
            cleaned_tokens.append(token)

    return ' '.join(cleaned_tokens)


def preprocess_dataframe(df, text_col='Resume_str', lemmatize=True):
    """Apply cleaning pipeline to entire dataframe."""
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(lambda x: clean_resume(x, lemmatize=lemmatize))
    df['clean_word_count'] = df['cleaned_text'].str.split().str.len()
    return df


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from config import DATA_PATH, CLEANED_CSV

    import pandas as pd
    DATA_PATH_LOCAL = DATA_PATH
    df = pd.read_csv(DATA_PATH_LOCAL)

    print("Testing preprocessing pipeline...")
    sample = df['Resume_str'].iloc[0]
    print("ORIGINAL (first 200 chars):")
    print(sample[:200])
    print("\nCLEANED:")
    print(clean_resume(sample)[:300])

    print("\nPreprocessing all resumes...")
    df_clean = preprocess_dataframe(df)
    print(f"  Done. Shape: {df_clean.shape}")
    print(f"  Avg cleaned word count: {df_clean['clean_word_count'].mean():.0f}")

    df_clean.to_csv(CLEANED_CSV, index=False)
    print(f"  Saved to {CLEANED_CSV}")

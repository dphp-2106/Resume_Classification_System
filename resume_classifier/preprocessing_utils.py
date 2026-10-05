"""
Shared preprocessing utilities — imported by all modules.
Ensures identical preprocessing during training and inference.
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

TECH_TOKENS = {
    'python', 'java', 'sql', 'aws', 'cpp', 'javascript', 'typescript',
    'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'sklearn',
    'nlp', 'ml', 'ai', 'tableau', 'powerbi', 'excel', 'sap', 'erp',
    'html', 'css', 'react', 'angular', 'nodejs', 'docker', 'kubernetes',
    'git', 'linux', 'agile', 'scrum', 'devops', 'ci', 'cd', 'api',
    'gcp', 'azure', 'hadoop', 'spark', 'kafka', 'mongodb', 'postgresql',
    'mysql', 'redis', 'elasticsearch', 'r', 'matlab', 'scala', 'swift',
    'kotlin', 'ruby', 'php', 'dotnet', 'blockchain', 'cybersecurity',
    'iot', 'ros', 'cad', 'autocad', 'solidworks', 'revit', 'photoshop',
    'illustrator', 'figma', 'pca', 'svm', 'lstm', 'gru',
    'bert', 'gpt', 'transformer', 'cnn', 'rnn', 'xgboost', 'lightgbm',
    'mba', 'bca', 'btech', 'mtech', 'phd', 'bsc', 'msc',
}


def clean_resume(text: str, lemmatize: bool = True) -> str:
    """
    Reproducible resume text cleaning pipeline.
    Must be called identically during training AND inference.
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

    # 5. Normalize C++, .NET, Node.js before lowercasing
    text = text.replace('C++', 'cpp').replace('c++', 'cpp')
    text = text.replace('.NET', 'dotnet').replace('.net', 'dotnet')
    text = text.replace('Node.js', 'nodejs').replace('node.js', 'nodejs')

    # 6. Lowercase
    text = text.lower()

    # 7. Remove non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 8. Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 9. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Tokenize, filter, lemmatize
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in TECH_TOKENS:
            cleaned_tokens.append(token)
            continue
        if token in STOP_WORDS:
            continue
        if lemmatize:
            token = lemmatizer.lemmatize(token)
        if len(token) >= 2:
            cleaned_tokens.append(token)

    return ' '.join(cleaned_tokens)

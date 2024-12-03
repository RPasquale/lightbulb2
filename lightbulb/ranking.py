# ranking.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy

def truncate_text(text, max_length=1024):
    tokens = text.split()
    if len(tokens) > max_length:
        return ' '.join(tokens[:max_length])
    return text

class RankingNN(nn.Module):
    def __init__(self, input_size=7):
        super(RankingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
ranking_model = RankingNN()
optimizer = optim.Adam(ranking_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()
scaler = MinMaxScaler()

# Download necessary resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")  # Small model to keep compute low

def preprocess_text(text):
    """
    Preprocess the input text by lowercasing, removing punctuation, and filtering out stopwords.
    Lemmatization is applied as well.
    """
    # Lowercase the text
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r'[' + string.punctuation + ']', ' ', text)

    # Tokenize the text into words
    words = word_tokenize(text)

    # Lemmatize, filter out stopwords and non-alphabetic words
    processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]

    return processed_words

def extract_named_entities(text):
    """
    Extract named entities (e.g., people, organizations, locations) from the text.
    """
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}]
    return named_entities

def extract_keywords_tfidf(corpus, text, n=5):
    """
    Extract keywords from the text using TF-IDF, combined with Named Entity Recognition and lemmatization.
    """
    # Preprocess the text and the entire corpus
    preprocessed_texts = [' '.join(preprocess_text(doc)) for doc in corpus]
    preprocessed_text = ' '.join(preprocess_text(text))

    # Named entities extraction
    named_entities = extract_named_entities(text)

    # Use TF-IDF vectorizer to find the most important words
    vectorizer = TfidfVectorizer(max_features=1000)  # Keep it light, max 1000 features
    X = vectorizer.fit_transform(preprocessed_texts)

    # Get the feature names (i.e., the words)
    feature_names = vectorizer.get_feature_names_out()

    # Transform the current text into TF-IDF scores
    response = vectorizer.transform([preprocessed_text])
    tfidf_scores = zip(feature_names, response.toarray()[0])

    # Sort by TF-IDF score
    sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

    # Combine top TF-IDF words with named entities for more richness
    keywords = [word for word, score in sorted_tfidf[:n]]
    combined_keywords = keywords + named_entities

    return combined_keywords[:n]

def extract_keywords(text, corpus, n=5):
    """
    Wrapper function that combines preprocessing, TF-IDF, and Named Entity Recognition to extract top N keywords.
    """
    if not text.strip():
        return []

    # Extract keywords using the TF-IDF based approach
    keywords = extract_keywords_tfidf(corpus, text, n)

    # If no meaningful keywords are found, fallback to keyword frequency
    if not keywords:
        return extract_fallback_keywords(text, n)

    return keywords

def extract_fallback_keywords(text, n=5):
    """
    Fallback method to extract keywords based on word frequency in case TF-IDF or NER fails.
    """
    words = preprocess_text(text)
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(n)]

def calculate_keyword_overlap(query_keywords, result_keywords):
    if len(query_keywords) == 0:
        return 0  # No keywords in query, so overlap is 0
    return len(set(query_keywords) & set(result_keywords)) / len(query_keywords)

def train_ranking_model(query, results, corpus=None, epochs=1):
    query = truncate_text(query)
    if not results:
        print("No results available. Skipping training.")
        return []

    if corpus is None:
        # If no corpus is provided, use results as a fallback
        corpus = [truncate_text(result['content']) for result in results if 'content' in result]

    query_embedding = transformer_model.encode(query)
    query_keywords = extract_keywords(query, corpus)

    training_data = []
    target_scores = []

    for result in results:
        # Truncate content
        content = truncate_text(result['content'])
        content_embedding = transformer_model.encode(content)
        
        # Handle missing 'title' and 'meta' fields with default values, and truncate
        title = truncate_text(result.get('title', ''))
        title_embedding = transformer_model.encode(title)
        
        meta_description = truncate_text(result.get('meta', {}).get('description', ''))
        meta_description_embedding = transformer_model.encode(meta_description)

        content_similarity = util.pytorch_cos_sim(query_embedding, content_embedding).item()
        title_similarity = util.pytorch_cos_sim(query_embedding, title_embedding).item()
        meta_description_similarity = util.pytorch_cos_sim(query_embedding, meta_description_embedding).item()

        # Handle missing metadata by providing default values
        content_length = result.get('meta', {}).get('content_length', 0)
        total_links = result.get('meta', {}).get('total_links', 0)

        result_keywords = extract_keywords(content, corpus)
        keyword_overlap = calculate_keyword_overlap(query_keywords, result_keywords)
        domain_authority = get_domain_authority(result.get('link', ''))

        features = [
            content_similarity, title_similarity, meta_description_similarity,
            content_length, total_links, keyword_overlap, domain_authority
        ]

        training_data.append(features)

        target_score = (0.4 * content_similarity + 0.3 * title_similarity + 
                        0.2 * meta_description_similarity + 0.1 * keyword_overlap)
        target_scores.append(target_score)

    # Normalize features
    training_data = scaler.fit_transform(training_data)
    training_data_tensor = torch.tensor(training_data, dtype=torch.float32)
    target_scores_tensor = torch.tensor(target_scores, dtype=torch.float32).unsqueeze(1)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_scores = ranking_model(training_data_tensor)
        loss = criterion(predicted_scores, target_scores_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Predict final scores and rank results
    with torch.no_grad():
        final_scores = ranking_model(training_data_tensor).squeeze().tolist()

    # Ensure final_scores is always a list
    if isinstance(final_scores, float):
        final_scores = [final_scores]

    for result, score in zip(results, final_scores):
        result['predicted_score'] = score

    ranked_results = sorted(results, key=lambda x: x['predicted_score'], reverse=True)
    return ranked_results

def get_domain_authority(url):
    # Placeholder function - replace with actual domain authority data if available
    high_authority_domains = ['arxiv.org', 'ncbi.nlm.nih.gov', 'nature.com', 'science.org']
    medium_authority_domains = ['wikipedia.org', 'stackexchange.com', 'github.com']
    
    for domain in high_authority_domains:
        if domain in url:
            return 1.0
    for domain in medium_authority_domains:
        if domain in url:
            return 0.7
    return 0.5
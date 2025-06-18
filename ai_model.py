from flask import Flask, request, jsonify
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import torch.nn.functional as F
import requests
from bs4 import BeautifulSoup
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Start flask
app = Flask(__name__)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Keyword extraction
def extract_keywords(text, num_keywords=10):
    # Tokenize text and remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]

    # Count word frequencies
    word_counts = Counter(tokens)

    # Return the most common keywords
    return [word for word, _ in word_counts.most_common(num_keywords)]

# Chunking text for large inputs
def chunk_text(text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    tokens = tokenizer.tokenize(text)

    # Split into chunks of size <= max_tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks

# Keyword check
def keyword_check(user_paragraph, source_keywords):
    return all(keyword in user_paragraph.lower() for keyword in source_keywords)

# Contradiction check
def contradiction_check(user_paragraph, source_paragraph):
    nli_model = pipeline("text-classification", model="roberta-large-mnli")  # Use roberta-large-mnli model
    user_chunks = chunk_text(user_paragraph)
    source_chunks = chunk_text(source_paragraph)

    for user_chunk in user_chunks:
        for source_chunk in source_chunks:
            result = nli_model(f"{source_chunk} [SEP] {user_chunk}")
            if any(res['label'] == 'CONTRADICTION' for res in result):
                return True
    return False

# Extract text content from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract all paragraph text
    paragraphs = soup.find_all('p')
    return " ".join(paragraph.text for paragraph in paragraphs)

# Select the best source from a list of trusted sources
def select_best_source(user_input, trusted_sources):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Compute the embedding of the user input
    user_encoded = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        user_output = model(**user_encoded)
    user_embedding = F.normalize(mean_pooling(user_output, user_encoded['attention_mask']), p=2, dim=1)

    best_similarity = -1  # Start with a low similarity
    best_source_url = None

    for source_url in trusted_sources:
        source_text = extract_text_from_url(source_url)
        source_chunks = chunk_text(source_text)  # Chunk the source text

        for source_chunk in source_chunks:
            source_encoded = tokenizer(source_chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                source_output = model(**source_encoded)
            source_embedding = F.normalize(mean_pooling(source_output, source_encoded['attention_mask']), p=2, dim=1)

            # Calculate similarity
            similarity = cosine_similarity(user_embedding, source_embedding).item()

            # Select the best source based on highest similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_source_url = source_url

    return best_source_url

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)

# Main function to check similarity between user content and selected source
def check_similarity(user_url, trusted_sources, num_keywords=10, threshold=0.9):
    # Extract user content from URL
    user_paragraph = extract_text_from_url(user_url)

    # Extract keywords from user content
    source_keywords = extract_keywords(user_paragraph, num_keywords)
    print(f"Extracted Keywords: {source_keywords}")

    # Select the best matching trusted source
    best_source_url = select_best_source(user_paragraph, trusted_sources)
    print(f"Selected trusted source: {best_source_url}")

    # Extract content from the selected source
    source_paragraph = extract_text_from_url(best_source_url)

    # Chunk text for model compatibility
    user_chunks = chunk_text(user_paragraph)
    source_chunks = chunk_text(source_paragraph)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    max_similarity = 0

    for user_chunk in user_chunks:
        encoded_input1 = tokenizer(user_chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output1 = model(**encoded_input1)
        embedding1 = F.normalize(mean_pooling(model_output1, encoded_input1['attention_mask']), p=2, dim=1)

        for source_chunk in source_chunks:
            encoded_input2 = tokenizer(source_chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output2 = model(**encoded_input2)
            embedding2 = F.normalize(mean_pooling(model_output2, encoded_input2['attention_mask']), p=2, dim=1)

            # Calculate similarity
            similarity = cosine_similarity(embedding1, embedding2).item()
            max_similarity = max(max_similarity, similarity)

    # Check conditions
    is_similar = max_similarity >= threshold
    has_keywords = keyword_check(user_paragraph, source_keywords)

    # Log contradictions separately
    has_contradictions = False
    try:
        has_contradictions = contradiction_check(user_paragraph, source_paragraph)
    except Exception as e:
        print(f"Contradiction check error: {e}")

    # Calculate the reliability percentage
    reliability_score = 0

    # Similarity score: higher similarity means higher reliability (max similarity = 1)
    reliability_score += max_similarity * 50  # Weight of 50%

    # Keyword match score: if keywords are matched, add 20%
    if has_keywords:
        reliability_score += 20

    # Contradiction check score: if there are contradictions, subtract 30%
    if not has_contradictions:
        reliability_score += 30  # Add 30% if no contradiction

    # Final reliability percentage
    reliability_score = min(reliability_score, 100)  # Cap it to 100%

    return round(reliability_score, 3)

@app.route('/')
def index():
    return 'Go to /ask for verifying reliability.'

@app.route('/ask', methods=['POST'])
@app.route('/ask', methods=['POST'])
def ask():
    # Trusted sources
    trusted_sources = [
        "https://apnews.com/",
        "https://www.reuters.com/",
        "https://www.bbc.com/news",
        "https://www.factcheck.org/",
    ]

    if request.method == 'POST':
        user_url = request.form.get('user_url')  # Get 'user_url' from form-data
        if not user_url:
            return jsonify({"error": "No URL provided"}), 400

        # Call check_similarity (returns a decimal number)
        similarity_score = check_similarity(user_url, trusted_sources)
        return jsonify({"similarity_score": similarity_score})  # Return as JSON
    else:
        return jsonify({"error": "Invalid request method"}), 400

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)

    # Run Flask app
    app.run(port=5000)

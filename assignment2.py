import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow as tf


# Function to preprocess a single document
def preprocess_document(document_content):
    # Extract the document number (DOCNO) from the document content
    doc_number = re.findall(r'<DOCNO>\s*([A-Za-z0-9-]+)\s*</DOCNO>', document_content)

    # Extract content between <TEXT> tags
    text_match = re.search(r'<TEXT>(.*?)</TEXT>', document_content, re.DOTALL)
    if text_match:
        text = text_match.group(1).strip()
    else:
        text = ""

    tokens = word_tokenize(text.lower()) 

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    stemmer = PorterStemmer()

    # Apply stemming to each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Return preprocessed tokens and document number
    return stemmed_tokens, doc_number

# Function to preprocess a query title and description
def preprocess_query(query):
    # Extract content between <title> and <desc>
    title_match = re.search(r'<title>(.*?)<desc>', query, re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = ""

    # Extract content between <desc> and </top>
    desc_match = re.search(r'<desc>(.*?)</top>', query, re.DOTALL)
    if desc_match:
        desc = desc_match.group(1).strip()
    else:
        desc = ""

    # Concatenate title and description
    query_text = title + " " + desc

    # Tokenize the lowercased text
    tokens = word_tokenize(query_text.lower())  # Convert to lowercase before tokenization

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Function to preprocess a query title and description
def preprocessing_query_title_and_desc(query):
    # Extract content between <title> and the end of the line or next tag
    title_match = re.search(r'<title>([^<\n]*)', query)
    title = title_match.group(1).strip() if title_match else ""

    # Extract content between <desc> and <narr>
    desc_match = re.search(r'<desc>(.*?)<narr>', query, re.DOTALL)
    desc = desc_match.group(1).strip() if desc_match else ""

    # Combine title and description
    query_text = title + " " + desc

    # Tokenize the lowercased text
    tokens = word_tokenize(query_text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # Apply stemming to each token
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens

# Function to retrieve documents for a given query
def retrieve_documents(query, inverted_index, document_tokens_dictionary):
    # Preprocess the query
    preprocessed_query = preprocess_query(query)

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Find documents containing at least one query term
    relevant_documents = set()
    for token in preprocessed_query:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])

    # Convert relevant documents to list
    relevant_documents = list(relevant_documents)
    
    document_tokens_back_to_text_array = []

    # Iterate over the documents and preprocess them
    for document_number, document_tokens in document_tokens_dictionary.items():
        document_tokens_back_to_text_array.append(' '.join(document_tokens))

    # Check if document_tokens_back_to_text_array is not empty
    if document_tokens_back_to_text_array:
        # Create a TfidfVectorizer
        vectorizer = TfidfVectorizer()

        # Fit the vectorizer to the document tokens
        document_tfidf_matrix = vectorizer.fit_transform(document_tokens_back_to_text_array)

        # Preprocess the query
        query_tokens_back_to_text = ' '.join(preprocessing_query_title_and_desc(query))

        # Transform the query tokens to TF-IDF vector
        query_tfidf_vector = vectorizer.transform([query_tokens_back_to_text])

    # Compute cosine similarity between query and documents
    cosine_similarities = cosine_similarity(query_tfidf_vector, document_tfidf_matrix)[0]

    # Combine document numbers and cosine similarities
    ranked_documents = sorted(zip(relevant_documents, cosine_similarities), key=lambda x: x[1], reverse=True)

    # Return ranked documents
    return ranked_documents[:1000]  # Retrieve top 1000 documents for each query

# Function to load documents from a folder
def load_documents(folder_path):
    documents = {}
    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            document_content = file.read()
            tokens, doc_number = preprocess_document(document_content)
            documents[doc_number[0]] = tokens
    return documents

# Function to load test queries from a file
def load_queries(file_path):
    with open(file_path, 'r') as file:
        queries = file.read().split('</top>')
    return queries

# Load documents
collection_folder = 'coll'
document_tokens_dictionary = load_documents(collection_folder)

# Load test queries
queries_file_path = 'topics1-50.txt'
test_queries = load_queries(queries_file_path)

# Initialize inverted index
inverted_index = {}

# Construct inverted index
for doc_number, tokens in document_tokens_dictionary.items():
    for token in tokens:
        if token not in inverted_index:
            inverted_index[token] = [doc_number]
        else:
            inverted_index[token].append(doc_number)

# Retrieve and print documents for each query
for query_id, query in enumerate(test_queries, start=1):
    print(f"Query {query_id}:")
    ranked_documents = retrieve_documents(query, inverted_index, document_tokens_dictionary)
    for rank, (doc_number, score) in enumerate(ranked_documents, start=1):
        print(f"Rank {rank}: Document {doc_number} - Score: {score}")
    print("\n")

# Step 2: Re-ranking with Advanced Neural Models

# Load BERT-based model
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to re-rank documents using BERT
def rerank_with_bert(query, ranked_documents, document_text_dictionary):
    reranked_documents = []
    
    # Tokenize the query
    query_tokens = tokenizer.encode(query, add_special_tokens=True, return_tensors='pt')
    query_tokens_list = tokenizer.convert_ids_to_tokens(query_tokens[0].tolist()) 
    query_text = tokenizer.convert_tokens_to_string(query_tokens_list)     
    # Get BERT model outputs
    outputs = bert_model.encode(query_text)  

    query_embedding = torch.tensor(outputs) if isinstance(outputs, np.ndarray) else outputs

    # Iterate over ranked documents and calculate similarity scores
    for doc_id, _ in ranked_documents:

        doc_text = document_text_dictionary.get(doc_id, "")
        doc_text = str(doc_text)
        doc_tokens = tokenizer.encode(doc_text, add_special_tokens=True, return_tensors='pt')
        
        # Get BERT model outputs for document
        doc_outputs = bert_model.encode(doc_text) 
        doc_embedding = torch.tensor(doc_outputs) if isinstance(doc_outputs, np.ndarray) else doc_outputs
        similarity_score = calculate_similarity_score(query_embedding, doc_embedding)
        reranked_documents.append((doc_id, similarity_score))
    reranked_documents.sort(key=lambda x: x[1], reverse=True)

    return reranked_documents


def calculate_similarity_score(query_embedding, doc_embedding):
    # Ensure query_embedding and doc_embedding are 2D arrays
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if len(doc_embedding.shape) == 1:
        doc_embedding = doc_embedding.reshape(1, -1)
    
    similarity_score = cosine_similarity(query_embedding, doc_embedding)
    print("Similarity score:", similarity_score)

    return similarity_score


# Load Universal Sentence Encoder model
use_model = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import SentenceTransformer

def rerank_with_use(query, ranked_documents):
    # Load Universal Sentence Encoder model
    use_model = SentenceTransformer('all-MiniLM-L6-v2')

    reranked_documents = []
    query_embedding = use_model.encode([query])[0]

    # Compute similarity scores for each document
    for doc_id, doc_text in ranked_documents:

        doc_text = str(doc_text)

        doc_embedding = use_model.encode([doc_text])[0]

        # Compute cosine similarity between query and document embeddings
        similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]

        reranked_documents.append((doc_id, similarity_score))

    reranked_documents.sort(key=lambda x: x[1], reverse=True)

    return reranked_documents

# Iterate over test queries and re-rank initial results
for query_id, query in enumerate(test_queries, start=1):
    print(f"Re-ranking for Query {query_id}:")
    
    # Initial retrieval step
    ranked_documents = retrieve_documents(query, inverted_index, document_tokens_dictionary)
    
    # Re-ranking with BERT
    reranked_results_bert = rerank_with_bert(query, ranked_documents, document_tokens_dictionary)
    
    # Re-ranking with USE
    reranked_results_use = rerank_with_use(query, ranked_documents)
    
    # Print re-ranked results
    print("BERT Re-ranking:")
    for rank, (doc_number, score) in enumerate(reranked_results_bert, start=1):
        print(f"Rank {rank}: Document {doc_number} - Score: {score}")
    
    print("\nUSE Re-ranking:")
    for rank, (doc_number, score) in enumerate(reranked_results_use, start=1):
        print(f"Rank {rank}: Document {doc_number} - Score: {score}")
    
    print("\n")

# Step 3: Output Results to File

# Function to write results to file
def write_results_to_file(results, file_path):
    with open(file_path, 'w') as file:
        for query_id, ranked_documents in enumerate(results, start=1):
            for rank, (doc_number, score) in enumerate(ranked_documents, start=1):
                file.write(f"{query_id} Q0 {doc_number} {rank} {score} group28\n")

results = []

# Iterate over test queries and store results
for query_id, query in enumerate(test_queries, start=1):
    ranked_documents = retrieve_documents(query, inverted_index, document_tokens_dictionary)
    results.append(ranked_documents)
    
results_file_path = 'Results.txt'
write_results_to_file(results, results_file_path)

import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from collections import defaultdict


def load_documents(folder_path):
    # Initialize an empty list to store the documents
    documents = []

    # Iterate over all files in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Read the content of each file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                document = file.read()
                documents.append(document)

    return documents


def preprocess(document, min_token_length=3, min_occurrences=3):
    # Tokenization
    tokens = word_tokenize(document.lower())

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Lancaster stemming
    lancaster = LancasterStemmer()
    tokens = [lancaster.stem(token) for token in tokens]

    # Count occurrences of each word using a dictionary
    word_counts = {}
    for token in tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1

    # Eliminate words that appear only once or twice
    tokens = [token for token in tokens if word_counts[token] > min_occurrences]

    # Filter by minimum token length and remove non-words
    tokens = [token for token in tokens if len(token) >= min_token_length]

    # Remove duplicates
    tokens = list(set(tokens))

    return tokens


def build_inverted_index(documents):
    inverted_index = defaultdict(list)

    for doc_id, document in enumerate(documents):
        processed_tokens = preprocess(document)

        for token in processed_tokens:
            inverted_index[token].append(doc_id)

    return inverted_index


def main():
    folder_path = "/Users/yara/Desktop/Winter 2024/CSI4107/A1_Group12/AP_collection/coll"

    # Call the load_documents function
    documents = load_documents(folder_path)
    total_token_count = 0

    # Build inverted index
    inverted_index = build_inverted_index(documents)

    # Display the results
    for word, postings in inverted_index.items():
        print(f"Word: {word}, Postings: {postings}")

    # Calculate total token count
    for doc_id, document in enumerate(documents):
        processed_tokens = preprocess(document)
        total_token_count += len(processed_tokens)

    print(f"Total Token Count Across All Documents: {total_token_count}")


if __name__ == "__main__":
    main()

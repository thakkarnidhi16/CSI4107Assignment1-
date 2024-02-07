import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import PorterStemmer

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

def preprocess(document):
    # Step 1: Tokenization
    tokens = word_tokenize(document.lower())  # Convert to lowercase for consistency
    
    # Step 2: Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    #Step 3: Porter stemming
    #porter = PorterStemmer()
    #tokens = [porter.stem(token) for token in tokens]
    
    return tokens

def main():
    
    folder_path = "/Users/Rajvir/OneDrive/Documents/y4s1/csi4107/CSI4107Assignment1-/coll"
    
    documents = load_documents(folder_path)
    
    doc_id = 0
    document = documents[doc_id]
    processed_tokens = preprocess(document)

    #for doc_id, document in enumerate(documents):
        # Preprocess each document
    #    processed_tokens = preprocess(document) 
        
        # Display the results
   # print(f"Document {doc_id + 1} - Original Text: {document[:50]}...")
    print(f"Document {doc_id + 1} - Processed Tokens: {processed_tokens}")
    print("=" * 50)
    print(len(processed_tokens))

if __name__ == "__main__":
    main()
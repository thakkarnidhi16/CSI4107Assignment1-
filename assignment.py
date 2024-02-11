import os
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from collections import defaultdict

def load_query():
    # Read the content of the file
    with open('query.txt', 'r') as file:
        content = file.read()

    # Split the content into lines
    lines = content.split('\n')

    # Extract titles from lines containing "<title>"
    titles = [line.strip()[7:] for line in lines if line.strip().startswith("<title>")]
    return titles

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



custom_stopwords = [
    'a', 'about', 'above', 'ac', 'according', 'accordingly', 'across', 'actually', 'ad', 'adj', 'af', 'after',
    'afterwards', 'again', 'against', 'al', 'albeit', 'all', 'almost', 'alone', 'along', 'already', 'als', 'also',
    'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'apart', 'apparently', 'are', 'aren', 'arise', 'around', 'as', 'aside', 'at',
    'au', 'auf', 'aus', 'aux', 'av', 'avec', 'away', 'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
    'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'begins', 'behind', 'bei', 'being', 'below',
    'beside', 'besides', 'best', 'better', 'between', 'beyond', 'billion', 'both', 'briefly', 'but', 'by', 'c',
    'came', 'can', 'cannot', 'canst', 'caption', 'captions', 'certain', 'certainly', 'cf', 'choose', 'chooses',
    'choosing', 'chose', 'chosen', 'clear', 'clearly', 'co', 'come', 'comes', 'con', 'contrariwise', 'cos', 'could',
    'couldn', 'cu', 'd', 'da', 'dans', 'das', 'day', 'de', 'degli', 'dei', 'del', 'della', 'delle', 'dem', 'den',
    'der', 'deren', 'des', 'di', 'did', 'didn', 'die', 'different', 'din', 'do', 'does', 'doesn', 'doing', 'don',
    'done', 'dos', 'dost', 'double', 'down', 'du', 'dual', 'due', 'durch', 'during', 'e', 'each', 'ed', 'eg', 'eight',
    'eighty', 'either', 'el', 'else', 'elsewhere', 'em', 'en', 'end', 'ended', 'ending', 'ends', 'enough', 'es',
    'especially', 'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'except',
    'excepts', 'excepted', 'excepting', 'exception', 'exclude', 'excluded', 'excludes', 'excluding', 'exclusive', 'f',
    'fact', 'facts', 'far', 'farther', 'farthest', 'few', 'ff', 'fifty', 'finally', 'first', 'five', 'foer', 'follow',
    'followed', 'follows', 'following', 'for', 'former', 'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'fra',
    'frequently', 'from', 'front', 'fuer', 'further', 'furthermore', 'furthest', 'g', 'gave', 'general', 'generally',
    'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'going', 'gone', 'good', 'got', 'great', 'greater',
    'h', 'had', 'haedly', 'half', 'halves', 'hardly', 'has', 'hasn', 'hast', 'hath', 'have', 'haven', 'having', 'he',
    'hence', 'henceforth', 'her', 'here', 'hereabouts', 'hereafter', 'hereby', 'herein', 'hereto', 'hereupon', 'hers',
    'herself', 'het', 'high', 'higher', 'highest', 'him', 'himself', 'hindmost', 'his', 'hither', 'how', 'however',
    'howsoever', 'hundred', 'hundreds', 'i', 'ie', 'if', 'ihre', 'ii', 'im', 'immediately', 'important', 'in', 'inasmuch',
    'inc', 'include', 'included', 'includes', 'including', 'indeed', 'indoors', 'inside', 'insomuch', 'instead', 'into',
    'inward', 'is', 'isn', 'it', 'its', 'itself', 'j', 'ja', 'journal', 'journals', 'just', 'k', 'kai', 'keep', 'keeping',
    'kept', 'kg', 'kind', 'kinds', 'km', 'l', 'la', 'large', 'largely', 'larger', 'largest', 'las', 'last', 'later', 'latter',
    'latterly', 'le', 'least', 'les', 'less', 'lest', 'let', 'like', 'likely', 'little', 'll', 'long', 'longer', 'los', 'low',
    'lower', 'lowest', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'making', 'many', 'may', 'maybe', 'me', 'meantime',
    'meanwhile', 'med', 'might', 'million', 'mine', 'miss', 'mit', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'ms',
    'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'nach', 'namely', 'nas', 'near', 'nearly', 'necessarily', 'necessary',
    'need', 'needs', 'needed', 'needing', 'neither', 'nel', 'nella', 'never', 'nevertheless', 'new', 'next', 'nine', 'ninety',
    'no', 'nobody', 'none', 'nonetheless', 'noone', 'nope', 'nor', 'nos', 'not', 'note', 'noted', 'notes', 'noting', 'nothing',
    'notwithstanding', 'now', 'nowadays', 'nowhere', 'o', 'obtain', 'obtained', 'obtaining', 'obtains', 'och', 'of', 'off',
    'often', 'og', 'ohne', 'ok', 'old', 'om', 'on', 'once', 'onceone', 'one', 'only', 'onto', 'or', 'ot', 'other', 'others',
    'otherwise', 'ou', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'par',
    'para', 'particular', 'particularly', 'past', 'per', 'perhaps', 'please', 'plenty', 'plus', 'por', 'possible', 'possibly',
    'pour', 'poured', 'pouring', 'pours', 'predominantly', 'previously', 'pro', 'probably', 'prompt', 'promptly', 'provide',
    'provides', 'provided', 'providing', 'q', 'quite', 'r', 'rather', 're', 'ready', 'really', 'recent', 'recently', 'regardless',
    'relatively', 'respectively', 'round', 's', 'said', 'same', 'sang', 'save', 'saw', 'say', 'second', 'see', 'seeing', 'seem',
    'seemed', 'seeming', 'seems', 'seen', 'sees', 'seldom', 'self', 'selves', 'send', 'sending', 'sends', 'sent', 'ses', 'seven',
    'seventy', 'several', 'shall', 'shalt', 'she', 'short', 'should', 'shouldn', 'show', 'showed', 'showing', 'shown', 'shows',
    'si', 'sideways', 'significant', 'similar', 'similarly', 'simple', 'simply', 'since', 'sing', 'single', 'six', 'sixty',
    'sleep', 'sleeping', 'sleeps', 'slept', 'slew', 'slightly', 'small', 'smote', 'so', 'sobre', 'some', 'somebody', 'somehow',
    'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'spake', 'spat', 'speek', 'speeks', 'spit',
    'spits', 'spitting', 'spoke', 'spoken', 'sprang', 'sprung', 'staves', 'still', 'stop', 'strongly', 'substantially', 'successfully',
    'such', 'sui', 'sulla', 'sung', 'supposing', 'sur', 't', 'take', 'taken', 'takes', 'taking', 'te', 'ten', 'tes', 'than', 'that',
    'the', 'thee', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'thenceforth', 'there', 'thereabout', 'thereabouts',
    'thereafter', 'thereby', 'therefor', 'therefore', 'therein', 'thereof', 'thereon', 'thereto', 'thereupon', 'these', 'they',
    'thing', 'things', 'third', 'thirty', 'this', 'those', 'thou', 'though', 'thousand', 'thousands', 'three', 'thrice', 'through',
    'throughout', 'thru', 'thus', 'thy', 'thyself', 'til', 'till', 'time', 'times', 'tis', 'to', 'together', 'too', 'tot', 'tou',
    'toward', 'towards', 'trillion', 'trillions', 'twenty', 'two', 'u', 'ueber', 'ugh', 'uit', 'un', 'unable', 'und', 'under',
    'underneath', 'unless', 'unlike', 'unlikely', 'until', 'up', 'upon', 'upward', 'us', 'use', 'used', 'useful', 'usefully',
    'user', 'users', 'uses', 'using', 'usually', 'v', 'van', 'various', 've', 'very', 'via', 'vom', 'von', 'voor', 'vs', 'w', 'want',
    'was', 'wasn', 'way', 'ways', 'we', 'week', 'weeks', 'well', 'went', 'were', 'weren', 'what', 'whatever', 'whatsoever', 'when',
    'whence', 'whenever', 'whensoever', 'where', 'whereabouts', 'whereafter', 'whereas', 'whereat', 'whereby', 'wherefore',
    'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wheresoever', 'whereto', 'whereunto', 'whereupon', 'wherever',
    'wherewith', 'whether', 'whew', 'which', 'whichever', 'whichsoever', 'while', 'whilst', 'whither', 'who', 'whoever', 'whole',
    'whom', 'whomever', 'whomsoever', 'whose', 'whosoever', 'why', 'wide', 'widely', 'will', 'wilt', 'with', 'within', 'without',
    'won', 'worse', 'worst', 'would', 'wouldn', 'wow', 'x', 'xauthor', 'xcal', 'xnote', 'xother', 'xsubj', 'y', 'ye', 'year', 'yes',
    'yet', 'yipee', 'you', 'your', 'yours', 'yourself', 'yourselves', 'yu', 'z', 'za', 'ze', 'zu', 'zum'
]


def is_valid_word(word):
    # Check if the word is a valid English word using WordNet
    synsets = wordnet.synsets(word)
    return len(synsets) > 0

def preprocess(document, min_token_length=3, min_occurrences=3):
    # Tokenization
    tokens = word_tokenize(document.lower())
    
    # Stopword removal
    stop_words = set(stopwords.words('english') + custom_stopwords)
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
    tokens = [token for token in tokens if len(token) >= min_token_length and is_valid_word(token)]
    
    # Remove duplicates
    tokens = list(set(tokens))
    
    return tokens

def compute_cosine_similarity(query_vector, document_vector):

    dot_product = sum(query_vector.get(term, 0) * document_vector.get(term, 0) for term in set(query_vector) & set(document_vector))
    query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
    doc_norm = math.sqrt(sum(val ** 2 for val in document_vector.values()))

    if query_norm == 0 or doc_norm == 0:
        return 0  # To avoid division by zero
    return dot_product / (query_norm * doc_norm)

def search(query, inverted_index, documents):
    query_terms = word_tokenize(query)
    query_vector = {term: query_terms.count(term) for term in set(query_terms)}

    scores = []
    for term, query_term_frequency in query_vector.items():
        if term in inverted_index:
            postings_list = inverted_index[term]
            for doc_id, document in enumerate(documents):  # Enumerate over documents to get doc_id
                document_terms = word_tokenize(document)
                document_vector = {term: inverted_index.get(term, [(doc_id, 0)])[0][1] for term in set(document_terms)}
                similarity = compute_cosine_similarity(query_vector, document_vector)
                scores.append((doc_id, similarity))

    # Combine scores for the same document and sort the results based on similarity score
    scores_dict = defaultdict(float)
    for doc_id, similarity in scores:
        scores_dict[doc_id] += similarity

    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_scores





#step2 indexing 
def build_inverted_index(documents):
    inverted_index = defaultdict(list)

    for doc_id, document in enumerate(documents):
        processed_tokens = preprocess(document)
        term_frequency = dict()

        for token in processed_tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        for term, frequency in term_frequency.items():
            inverted_index[term].append((doc_id, frequency))

    return inverted_index

def main():
    folder_path = "/Users/Rajvir/OneDrive/documents/y4s1/csi4107/csi4107assignment1-/AP_collection/coll"

    # Call the load_documents function
    documents = load_documents(folder_path)
    total_token_count = 0

    # Build inverted index
    inverted_index = build_inverted_index(documents)

    # # Display the results
    # for word, postings in inverted_index.items():
    #     print(f"Word: {word}, Postings: {postings}")

    queries = load_query()
    results = search("sleepless crossword saddest christmases horst marengo redeploy", inverted_index, documents)
    print("Search Results:")
    for doc_id, similarity in results:
        print(f"Document {doc_id + 1}: Similarity = {similarity:.4f}")

    # Calculate total token count
    """ for doc_id, document in enumerate(documents):
        processed_tokens = preprocess(document)
        total_token_count += len(processed_tokens) """

    #print(f"Total Token Count Across All Documents: {total_token_count}")


if __name__ == "__main__":
    main()

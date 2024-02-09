import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter


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


#step2 indexing 
def build_inverted_index(documents):
    inverted_index = {}

    for doc_id, document in enumerate(documents):
        processed_tokens = preprocess(document)

        for token in processed_tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append(doc_id)

    return inverted_index

def main():

    
    folder_path = "C:/Users/thakk/Desktop/uottawa/Winter2024/CSI4107/Assignment1/AP_collection/coll"

    

    # Call the load_documents function
    documents = load_documents(folder_path)
    total_token_count = 0
    doc_id = 0
    document = documents[doc_id]
    processed_tokens = preprocess(document)

    for doc_id, document in enumerate(documents):
        #preprocess each document
        processed_tokens = preprocess(document) 
        total_token_count += len(processed_tokens)
        
        # Display the results
        print(f"Document {doc_id + 1} - Token Count: {len(processed_tokens)}")
        print(f"Document {doc_id + 1} - Original Text: {document[:50]}...")
        #print(f"Document {doc_id + 1} - Processed Tokens: {processed_tokens}")
        print("=" * 50)
        
    print(f"Total Token Count Across All Documents: {total_token_count}")


if __name__ == "__main__":
    main()
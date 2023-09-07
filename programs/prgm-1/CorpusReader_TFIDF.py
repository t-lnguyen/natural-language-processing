from nltk import *
import nltk.stem.snowball
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from math import log
class CorpusReader_TFIDF:
    def __init__(self, corpus, tf = "raw", idf = "base", stopWord = "none", 
                 toStem = False, stemFirst = False, ignoreCase = True):
        """
        corpus: NLTK corpus object
        tf: the method used to calculate term frequency. The following values are supported
            "raw": (default)=rawtermfrequency
            "log": log normalized (1 + log (frequency) if frequency > 0; 0 otherwise)
        idf: the method used to calculate the inverse document frequency
            "base" (default): basic inverse document frequency
            "smooth": inversefrequencysmoothed
        stopWord: what stopWords to remove
            “none” : no stopwords need to be removed
            “standard”: use the standard English stopWord available in NLTK
            Others: this should treat as a filename where stopwords are to be read. 
            You should assume any word inside the stopwords file is a stopword.
        toStem: if true, use the Snowball stemmer to stem the words beforehand
        stemFirst: if stopwords are used and stemming is set to yes (otherwise this flag is ignored), 
            then true means you stem before you remove stopwords, 
            and false means you remove stopwords before you stem
        ignoreCase: if true, ignore the case of the word (i.e. “Apple”, “apple”, “APPLE” are the same word). 
            In such case, represent the word as the all lower-case version (this include the words in the stopWord file). 
            Also, you will change all words into lowercase before you do any subsequent processing (e.g. remove stopwords and stemming)

        """
        self.corpus = corpus
        self.tf = tf
        self.idf_method = idf        
        self.stopWord = stopWord
        # stopWord instantiation logic
        if self.stopWord == "none":
            self.stopWords = set()
        elif self.stopWord == "standard":
            self.stopWords = set(stopwords.words('english'))
        else:
            with open(self.stopWord, "r") as stopWordFile:
                self.stopWords = set(stopWordFile.read().split(' '))
        self.toStem = toStem
        self.stemFirst = stemFirst
        self.ignoreCase = ignoreCase

        ## fields for calculating TF-IDF when instantiating the class
        self.stemmer = SnowballStemmer("english") if self.toStem else None
        self.tfidf_vector = []
        self.tf_values = []
        self.idf_values = []
        # file ids
        self.fileids = corpus.fileids()
        self.distinct_words = []
        # dictionary for distinct words and their index
        self.distinct_words_map = {}
        # store corpus after stemming and stopword removal
        self.filtered_corpus = {}
        self._preprocess()

    def _preprocess(self):
        # loop through the corpus's documents and apply 
        # cleaning and/or stemming
        words_count = 0
        docs_count = len(self.fileids)
        filtered_words = {}
        # instantiate dictionary for filtered corpi
        for fileid in self.fileids:
            self.filtered_corpus[fileid] = {}

        # loop through our corpus's documents
        for fileid in self.fileids:
            # loop through our document's words
            for word in self.words(fileid):
                norm_word = word
                # if unchanged word is in our filtered dictionary
                if norm_word in filtered_words:
                    word = filtered_words[norm_word]
                else:
                    # check case flag
                    if self.ignoreCase:
                        word = word.lower()
                    # check stemming flag
                    if self.toStem:
                        word = self.stemmer.stem(word)
                    # add our filtered word to the filtered dictionary
                    filtered_words[norm_word] = word
                # skip word if in stopWords list
                if word in self.stopWord:
                    continue
                # increment frequency of word for current document
                if word in self.filtered_corpus[fileid]:
                    self.filtered_corpus[fileid][word] += 1
                else:
                    # start the frequency of word increment
                    self.filtered_corpus[fileid][word] = 1
                # if we discover a unique word
                if word not in self.distinct_words_map:
                    # keep track of our unique word/term frequency
                    self.distinct_words_map[word] = words_count       
                    self.distinct_words.append(word)
                    words_count += 1
        ## outputs cleaned dictionary of unique words
        # calculate tf
        # calculate idf

    def fields(self):
        """ Returns the files of the corpus
        """
        return self.corpus.fields()
    
    def raw(self, fileids = None):
        """ Returns the raw content of the specified files
        """
        return self.corpus.raw(fileids)
    
    def words(self, fileids = None):
        """ Returns the words of the specified fileids
        """
        return self.corpus.words(fileids)        

    def calculate_tf(self, words):
        tf_vals = {}

        
        return dict(FreqDist(words))

    def calculate_idf(self):
        NotImplementedError()





    def tfidf(self, fileid, returnZero = False) -> dict:
        """ Return the TF-IDF for the specific document in the corpus (specified by fileid). 
        The vector is represented by a dictionary/hash in python. 
        The keys are the terms, and the values are the tf-idf value of the dimension. 
        fileid: specific document in the corpus
        returnZero: if returnZero is true, 
        then the dictionary will contain terms that have 0 value for that vector, 
        otherwise the vector will omit those terms
        """
        tf_idf_vector_dict = {}
        return tf_idf_vector_dict
    
    def tfidfAll(returnZero = False) -> dict:
        """ Return the TF-IDF for all documents in the corpus. 
        It will be returned as a dictionary. 
        The key is the fileid of each document, 
        for each document the value is the tfidf of that document (using the same format as above).
        returnZero: if returnZero is true, 
        then the dictionary will contain terms that have 0 value for that vector, 
        otherwise the vector will omit those terms
        """
        NotImplementedError()

    def tfidfNew(words: list):
        """ Return the tf-idf of a “new” document, represented by a list of words. 
        You should honor the various parameters (ignoreCase, toStem etc.) when preprocessing the new document. 
        Also, the idf of each word should not be changed 
        (i.e. the “new” document should not be treated as part of the corpus).
        words: list of words of a document
        """
        NotImplementedError()

    def idf():
        """ Return the idf of each term as 
        a dictionary : keys are the terms, and values are the idf

        """
        NotImplementedError()
    
    def cosine_sim(documents: list):
        """ Return the cosine similarity between two documents in the corpus
        documents: list of 2 documents to be compared via cosine similarity
        """
        NotImplementedError()
    
    def cosine_sim_new(words: list, fileid: str):
        """ Return the cosine similary between a “new” document 
        (as if specified like the tfidf_new() method) and the documents specified by fileid.
        words: list of words of a document
        fileid: specific document in the corpus
        """
        NotImplementedError()

    def query(words: list):
        """ BONUS Return a list of (document, cosine_sim) tuples 
        that calculate the cosine similarity between the “new” document 
        (specified by the list of words as the document). 
        The list should be ordered in decreasing order of cosine similarity.
        words: list of words of a document
        """
        NotImplementedError()
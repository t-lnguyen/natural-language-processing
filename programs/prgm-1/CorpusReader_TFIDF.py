from nltk import *
from nltk import FreqDist
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
        # file ids or our corpus's documents
        self.fileids = corpus.fileids()
        self.distinct_words = []
        # dictionary for distinct words and their index
        self.distinct_words_map = {}
        # store corpus after stemming and stopword removal
        self.filtered_corpus = {}
        self._preprocess()

    def _preprocess(self) -> None:
        """ processes the corpus's documents through cleaning/filtering
        and stemming, if enabled, to calculate TF, IDF, and TF-IDF
        of its documents
        """
        # cleaning and/or stemming
        docs_count = len(self.fileids)
        words_count = self._corpus_clean_filter()
        # calculate tf
        # list to track our document's non-zero index
        corpus_non_zero_indices = []
        self._process_tf(words_count=words_count, corpus_non_zero_indices=corpus_non_zero_indices)
        
        # calculate idf
        self._process_idf(words_count=words_count)
        
        # calculate tf-idf
        self._process_tf_idf(words_count=words_count, docs_count=docs_count,
                              corpus_non_zero_indices=corpus_non_zero_indices)

    def _process_tf_idf(self, words_count, docs_count, corpus_non_zero_indices) -> None:
        """ Produces a list of TF-IDF values for our entire corpus' documents.
            Applies logic for TF and IDF calculation variants 
        """
        for doc in range(docs_count):
            tf_idf_vector = [0] * words_count
            for corpus_non_zero_index in corpus_non_zero_indices[doc]:
                tf = self.tf_values[doc][corpus_non_zero_index]
                idf = self.idf_values[corpus_non_zero_index]

                if self.tf == "log" and tf != 0:
                    tf = 1 + log(tf, 2)

                if self.idf == "base":
                    idf = log(docs_count / float(idf), 2)
                elif self.idf == "smoothed":
                    idf = log(docs_count / (1 + float(idf)), 2)
                
                tf_idf_vector[corpus_non_zero_index] = tf * idf
            self.tfidf_vector.append(tf_idf_vector)

    def _process_idf(self, words_count) -> None:
        """ Processes corpus documents based on our filtered corpus to produce
            a list containing the count of how many documents contain each word.
        """
        self.idf_values = [0] * words_count
        # count how many documents contain each word
        for fileid in self.fileids:
            for word in self.filtered_corpus[fileid].keys():
                self.idf_values[self.distinct_words_map[word]] += 1
    
    def _process_tf(self, words_count, corpus_non_zero_indices = list) -> None:
        """ Processes corpus documents to produce a list of Term Frequency values
            utilizing FreqDist on our filtered corpus
        """
        for fileid in self.fileids:
            cur_doc_non_zero_indices = []
            cur_doc_tf_vector = [0] * words_count
            # utlize nltk FreqDist module to get the term:frequency pair for a doc
            cur_doc_tf_dist = FreqDist(self.filtered_corpus[fileid].keys())
            # iterate through our term:frequency pair
            for word, frequency in cur_doc_tf_dist.items():
                cur_doc_tf_vector[self.distinct_words_map[word]] = frequency
                if frequency > 0:
                    cur_doc_non_zero_indices.append(self.distinct_words_map[word])
            corpus_non_zero_indices.append(cur_doc_non_zero_indices)
            self.tf_values.append(cur_doc_tf_vector)

    def _corpus_clean_filter(self) -> int:
        """ Implement applicable case logic, stemmer logic and stopWord removal logic.
            Produces cleaned and filter corpus dictionary 
            and list of distinct words.
            Returns: count of distinct words
        """
        words_count = 0
        filtered_words = {}
        # instantiate our dictionary for filtered corpus documents
        for fileid in self.fileids:
            self.filtered_corpus[fileid] = {}
        # loop through our corpus's documents
        for fileid in self.fileids:
            # loop through our document's words
            for word in self.words(fileid):
                norm_word = word
                # check if unfiltered word is in our filtered dictionary
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
                    # keep track of our unique word and their index within the corpus documents
                    ## we consider the corpus documents as one whole document
                    self.distinct_words_map[word] = words_count       
                    words_count += 1
                    self.distinct_words.append(word)
        return words_count

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

    def tfidf(self, fileid, returnZero = False) -> dict:
        """ Return the TF-IDF for the specific document in the corpus (specified by fileid). 
        The vector is represented by a dictionary/hash in python. 
        The keys are the terms, and the values are the tf-idf value of the dimension. 
        fileid: specific document in the corpus
        returnZero: if returnZero is true, 
        then the dictionary will contain terms that have 0 value for that vector, 
        otherwise the vector will omit those terms
        """
        doc_tf_idf_result = {}
        doc_tf_idf_values = self.tfidf_vector[self.fileids.index(fileid)]
        # iterate through the document's distinct words and their indices
        for word, index in self.distinct_words_map.items():
            doc_tf_idf_value = doc_tf_idf_values[index]
            # check whether to omit terms of 0 TF-IDF
            if not returnZero and doc_tf_idf_value == 0:
                continue
            doc_tf_idf_result[word] = doc_tf_idf_value

        return doc_tf_idf_result
    
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
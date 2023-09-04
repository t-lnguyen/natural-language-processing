from nltk import *
class CorpusReader_TFIDF(corpus, tf = "raw", idf = "base", stopWord = "none", 
                         toStem = False, stemFirst = False, ignoreCase = True):

    def tfidf(fileid, returnZero = False):
        """ Return the TF-IDF for the specific document in the corpus (specified by fileid). 
        The vector is represented by a dictionary/hash in python. 
        The keys are the terms, and the values are the tf-idf value of the dimension. 
        If returnZero is true, 
        then the dictionary will contain terms that have 0 value for that vector, 
        otherwise the vector will omit those terms

        """
        NotImplementedError()
    
    def tfidfAll(returnZero = False):
        """ Return the TF-IDF for all documents in the corpus. 
        It will be returned as a dictionary. 
        The key is the fileid of each document, 
        for each document the value is the tfidf of that document (using the same format as above).

        """
        NotImplementedError()

    def tfidfNew(words: list):
        """ Return the tf-idf of a “new” document, represented by a list of words. 
        You should honor the various parameters (ignoreCase, toStem etc.) when preprocessing the new document. 
        Also, the idf of each word should not be changed 
        (i.e. the “new” document should not be treated as part of the corpus).

        """
        NotImplementedError()

    def idf():
        """ Return the idf of each term as 
        a dictionary : keys are the terms, and values are the idf

        """
        NotImplementedError()
    
    def cosine_sim(documents: list):
        """ Return the cosine similarity between two documents in the corpus
        
        """
        NotImplementedError()
    
    def cosine_sim_new(words: list, fileid: str):
        """ Return the cosine similary between a “new” document 
        (as if specified like the tfidf_new() method) and the documents specified by fileid.

        """
        NotImplementedError()

    def query(words: list):
        """ BONUS Return a list of (document, cosine_sim) tuples 
        that calculate the cosine similarity between the “new” document 
        (specified by the list of words as the document). 
        The list should be ordered in decreasing order of cosine similarity.
        
        """
        NotImplementedError()
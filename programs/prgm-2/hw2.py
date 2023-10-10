import gensim.downloader as g1

def simValues(model, key: str, countList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countList: a list of integers specifying the neighbor positions to retrieve
    Returns: 
        A list of similarity values for the m-th nearest neighbors. If an item in
        countList is not an integer or is out of range, the corresponding value is -10000.0.
    """
    NotImplementedError()

def simValuesPct(model, key: str, countPctList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countPctList: a list of integers specifying the percentiles to retrieve similarity values for.
    Returns:
    """
    NotImplementedError()
import gensim.downloader as g1

def simValues(model: g1.Word2VecKeyedVectors, key: str, countList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countList: a list of integers specifying the neighbor positions to retrieve
    Returns: 
        A list of similarity values for the m-th nearest neighbors. If an item in
        countList is not an integer or is out of range, the corresponding value is -10000.0.
    """
    if not isinstance(model, g1.Word2VecKeyedVectors):
        raise ValueError("The 'model' parameter must be a Gensim Word2Vec model.")

    if not isinstance(key, str):
        raise ValueError("The 'key' parameter must be a string.")

    if not all(isinstance(count, int) for count in countList):
        raise ValueError("All elements in 'countList' must be integers.")
    
    similarity_values = []

    if key not in model:
        # If the key is not in the vocabulary, return -10000.0 for all counts.
        return [-10000.0] * len(countList)

    for count in countList:
        if count <= 0 or count > len(model.vocab) - 1:
            # If count is out of range, return -10000.0.
            similarity_values.append(-10000.0)
        else:
            # Find the m-th nearest neighbor and calculate its similarity value.
            most_similar = model.most_similar(key, topn=count)
            similarity_values.append(most_similar[-1][1])  # Get the similarity value.

    return similarity_values

def simValuesPct(model: g1.Word2VecKeyedVectors, key: str, countPctList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countPctList: a list of integers specifying the percentiles to retrieve similarity values for.
    Returns:
    """
    NotImplementedError()
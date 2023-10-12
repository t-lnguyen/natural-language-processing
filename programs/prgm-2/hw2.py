from gensim.models import KeyedVectors

def simValues(model: KeyedVectors, key: str, countList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countList: a list of integers specifying the neighbor positions to retrieve
    Returns: 
        A list of similarity values for the m-th nearest neighbors. If an item in
        countList is not an integer or is out of range, the corresponding value is -10000.0.
    """
    if not isinstance(model, KeyedVectors):
        raise ValueError("The 'model' parameter must be a Gensim Word2Vec model.")

    if not isinstance(key, str):
        raise ValueError("The 'key' parameter must be a string.")
    
    if key not in model.key_to_index:
        # If the key is not in the vocabulary, return -10000.0 for all counts.
        return [-10000.0] * len(countList)
    
    similarity_values = []
    
    for count in countList:
        if not isinstance(count, int) or count <= 0 or count > len(model.index_to_key) - 1:
            # If count is out of range, return -10000.0.
            similarity_values.append(-10000.0)
        else:
            # Find the m-th nearest neighbor and calculate its similarity value.
            most_similar = model.most_similar(key, topn=count)
            similarity_values.append(most_similar[-1][1])  # Get the similarity value.

    return similarity_values

def simValuesPct(model: KeyedVectors, key: str, countPctList: []):
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countPctList: a list of integers specifying the percentiles to retrieve similarity values for.
    Returns:
    """
    if not isinstance(model, KeyedVectors):
        raise ValueError("The 'model' parameter must be a Gensim Word2Vec model.")

    if not isinstance(key, str):
        raise ValueError("The 'key' parameter must be a string.")
    
    if key not in model.key_to_index:
        # If the key is not in the vocabulary, return -10000.0 for all percentiles.
        return [-10000.0] * len(countPctList)

    similarity_values = []

    for countPct in countPctList:
        if countPct < 0 or countPct > 100:
            # If countPct is out of range, return -10000.0.
            similarity_values.append(-10000.0)
        elif countPct == 0:
            # if the value is 0, it should return the similarity of the most similar word.
            similarity_values.append(model.most_similar(key, topn=1)[0][1])
        else:
            # Calculate the percentile index and retrieve the similarity value.
            percentile_index = int(countPct * (len(model.index_to_key) - 1) / 100)
            
            similarity_values.append(model.similarity(w1=key, w2=model.index_to_key[percentile_index]))

    return similarity_values

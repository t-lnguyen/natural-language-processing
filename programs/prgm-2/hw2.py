from gensim.models import KeyedVectors
from nltk.corpus.reader.wordnet import Synset
from numpy import mean, std

def simValues(model: KeyedVectors, key: str, countList: []) -> list:
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

def simValuesPct(model: KeyedVectors, key: str, countPctList: []) -> list:
    """ 
    Args:
        model: a Word2Vec model loaded from Gensim
        key: target word
        countPctList: a list of integers specifying the percentiles to retrieve similarity values for.
    Returns:
        A list of similarity values for the specified percentiles. If an item in countPctList is not an integer
        or is out of range, the corresponding value is -10000.0.
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

def synsetSimValue(model: KeyedVectors, synset: Synset) -> list:
    """"
    Args:
        model: a Word2Vec model loaded from Gensim
        synset: a WordNet synset
    Returns:
        - A list of four numbers: [avg, sd, min, max] representing average similarity, standard deviation,
        minimum similarity, and maximum similarity between words in the synset.
        - If the synset contains 0 or 1 word, an empty list.
        - If the synset contains 2 words, the std dev is returned as 0.
    """
    if not isinstance(model, KeyedVectors):
        raise ValueError("The 'model' parameter must be a Gensim Word2Vec model.")
    if not isinstance(synset, Synset):
        raise ValueError("The 'synset' parameter must be a WordNet Synset synset")
    
    words_in_synset = [lemma.name() for lemma in synset.lemmas() if "_" not in lemma.name()]
    
    if len(words_in_synset) <= 1:
        return [] if not words_in_synset else [0, 0, 0, 0]
    
    # Calculate similarity values for all pairs of words in the synset.
    similarities = []
    for i in range(len(words_in_synset)):
        for j in range(i + 1, len(words_in_synset)):
            similarity = model.similarity(words_in_synset[i], words_in_synset[j])
            similarities.append(similarity)

    avg_simil = mean(similarities)
    std_dev = std(similarities)
    min_simil = min(similarities)
    max_simil = max(similarities)

    return [avg_simil, std_dev, min_simil, max_simil]
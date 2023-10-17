from hw2 import simValues, simValuesPct, synsetSimValue
import gensim.downloader as g1
from nltk.corpus import wordnet as wn
from nltk import download
import matplotlib.pyplot as plt

# word: the word to be searched
# pos: the part of speech 
## n: noun
## a: adjective
## v: verb
## r: adverb
# nn: which definition to be use
## must refer to wordnetweb.princeton.edu
SYNSET_ADVERB = "all.r.01"
SYNSET_NOUN = "person.n.01"
SYNSET_VERB = "be.v.07"


download('wordnet')
model = g1.load("glove-wiki-gigaword-100")
synset = wn.synset(SYNSET_VERB)


def part1():
    l1 = simValues(model=model, key="board", countList=[1, 5, 50, 10])
    print(l1)

    l2 = simValuesPct(model=model, key="board", countPctList=[0, 25, 50, 75, 100])
    print(l2)


    #======== Getting Distribution of most, 5th, and 10th similar word values ========#
    words_eval = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "whiskey",
        "x-ray",
        "yankee",
        "zulu"
    ]

    for word in words_eval:
        print(f"Word: {word}")
        most_similar = simValues(model, word, [1])
        fifth_similar = simValues(model, word, [5])
        tenth_similar = simValues(model, word, [10])

        print(f"Most Similar: {most_similar[0]}")
        print(f"Fifth Most Similar: {fifth_similar[0]}")
        print(f"Tenth Most Similar: {tenth_similar[0]}")
        print()

    most_similar_values = []
    fifth_similar_values = []
    tenth_similar_values = []

    for word in words_eval:
        most_similar = simValues(model, word, [1])
        fifth_similar = simValues(model, word, [5])
        tenth_similar = simValues(model, word, [10])

        most_similar_values.append(most_similar[0])
        fifth_similar_values.append(fifth_similar[0])
        tenth_similar_values.append(tenth_similar[0])

    # Analyze and visualize the distribution of similarity values here.
    # You can use NumPy for statistical analysis and Matplotlib for visualization.

    # For example, you can create a histogram for each set of values.
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(131)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(most_similar_values, bins=30)
    plt.title("Most Similar")

    ax = plt.subplot(132)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(fifth_similar_values, bins=30)
    plt.title("Fifth Most Similar")

    ax = plt.subplot(133)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(tenth_similar_values, bins=30)
    plt.title("Tenth Most Similar")

    plt.show()

    for word in words_eval:
        print(f"Word: {word}")
        most_similar = simValuesPct(model, word, [1])
        fifth_similar = simValuesPct(model, word, [5])
        tenth_similar = simValuesPct(model, word, [10])

        print(f"Most Similar: {most_similar[0]}")
        print(f"Fifth Most Similar: {fifth_similar[0]}")
        print(f"Tenth Most Similar: {tenth_similar[0]}")
        print()

    most_similar_values = []
    fifth_similar_values = []
    tenth_similar_values = []

    for word in words_eval:
        most_similar = simValuesPct(model, word, [1])
        fifth_similar = simValuesPct(model, word, [5])
        tenth_similar = simValuesPct(model, word, [10])

        most_similar_values.append(most_similar[0])
        fifth_similar_values.append(fifth_similar[0])
        tenth_similar_values.append(tenth_similar[0])

    # Analyze and visualize the distribution of similarity values here.
    # You can use NumPy for statistical analysis and Matplotlib for visualization.

    # For example, you can create a histogram for each set of values.
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(131)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(most_similar_values, bins=30)
    plt.title("Most Similar")

    ax = plt.subplot(132)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(fifth_similar_values, bins=30)
    plt.title("Fifth Most Similar")

    ax = plt.subplot(133)
    ax.set_xlabel("Percentage of Similarity")
    ax.set_ylabel("Count of Similarity Values")
    plt.hist(tenth_similar_values, bins=30)
    plt.title("Tenth Most Similar")

    plt.show()


def part2():
    #TODO revisit lecture on synsets -> determine "depth", "hypernym-hyponym" topics
    stats = synsetSimValue(model=model, synset=synset)
    print(f"Average Similiarity: {stats[0]}")
    print(f"Standard Deviation: {stats[0]}")
    print(f"Min Similiarity: {stats[0]}")
    print(f"Max Similiarity: {stats[0]}")

    # analyze relationship between depth in WordNet hierarchy and similarity changes
    # analyze relationship between words in the same synset and different synsets 
    # with respect to similarity values
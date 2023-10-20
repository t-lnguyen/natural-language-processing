from hw2 import simValues, simValuesPct, synsetSimValue, generate_words_in_synset
import gensim.downloader as g1
from nltk.corpus import wordnet as wn
from nltk import download
from numpy import mean
import matplotlib.pyplot as plt

import random
import os

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
synset = wn.synset(SYNSET_NOUN)
random.seed(42)

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
    hw2_path = os.path.dirname(__file__)
    with open(f"{hw2_path}/words.txt", mode="r") as words_file:
        words_eval = words_file.read().splitlines()

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

        most_similar_values.append((word, most_similar[0]))
        fifth_similar_values.append((word, fifth_similar[0]))
        tenth_similar_values.append((word, tenth_similar[0]))

    with open("most_sim_values.csv", mode="w") as most_file:
        for sim_val in list(dict.fromkeys(most_similar_values)):
            most_file.write(f"{sim_val[0]},{sim_val[1]}\n")
    with open("5th_most_sim_values.csv", mode="w") as most_file:
        for sim_val in list(dict.fromkeys(fifth_similar_values)):
            most_file.write(f"{sim_val[0]},{sim_val[1]}\n")
    with open("tenth_most_sim_values.csv", mode="w") as most_file:
        for sim_val in list(dict.fromkeys(tenth_similar_values)):
            most_file.write(f"{sim_val[0]},{sim_val[1]}\n")
    # plt.show()

def part2():
    stats = synsetSimValue(model=model, synset=synset)
    print(f"Average Similiarity: {stats[0]}")
    print(f"Standard Deviation: {stats[1]}")
    print(f"Min Similiarity: {stats[2]}")
    print(f"Max Similiarity: {stats[3]}")

    # analyze relationship between depth in WordNet hierarchy and similarity changes
    depths = []
    avg_similarities = []
    avg_std_devs = []
    for hypernym in synset.hypernyms():
        depth = max(len(hyp_path) for hyp_path in synset.hypernym_paths())
        similarities = []
        std_devs = []
        for hyponym in hypernym.hyponyms():
            stats = synsetSimValue(model, hyponym)
            if stats:
                similarities.append(stats[0])
                std_devs.append(stats[1])
        avg_similarity = mean(similarities)
        avg_std_dev = mean(std_devs)
        depths.append(depth)
        avg_similarities.append(avg_similarity)
        avg_std_devs.append(avg_std_dev)

    print(f"Depths of Hypernyms: {depths}")
    print(f"Average Similarities at Different Depths: {avg_similarities}")
    print(f"Average Standard Deviations at Different Depths: {avg_std_dev}")

    # analyze relationship between words in the same synset and different synsets 
    # with respect to similarity values
    words_in_synset = generate_words_in_synset(synset=synset)
    some_word = random.choice(words_in_synset)
    same_synset_sim_vals = [
        (some_word, word, model.similarity(some_word, word))
        for word in words_in_synset
    ]
    
    some_different_synset = wn.synset('man.n.11')
    blah = generate_words_in_synset(synset=some_different_synset)
    some_different_synset_sim_vals = [
        (some_word, word, model.similarity(some_word, word))
        for word in generate_words_in_synset(synset=some_different_synset)
    ]

    print(f"Similarities within the Same Synset: {same_synset_sim_vals}")
    print(f"Similarities with a Different Synset: {some_different_synset_sim_vals}")

    # analyze how the distance between synsets has an effect on the previous findings
part1()
#part2()
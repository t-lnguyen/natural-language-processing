from hw2 import simValues, simValuesPct
import gensim.downloader as g1
import numpy as np
import matplotlib.pyplot as plt

model = g1.load("glove-wiki-gigaword-100")

# l1 = simValues(model=model, key="board", countList=[1, 5, 50, 10])

# print(l1)

# l2 = simValuesPct(model=model, key="board", countPctList=[0, 25, 50, 75, 100])

# print(l2)


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
ax.set_ylabel("Count of Similarty Values")
plt.hist(most_similar_values, bins=30)
plt.title("Most Similar")

ax = plt.subplot(132)
ax.set_xlabel("Percentage of Similarity")
ax.set_ylabel("Count of Similarty Values")
plt.hist(fifth_similar_values, bins=30)
plt.title("Fifth Most Similar")

ax = plt.subplot(133)
ax.set_xlabel("Percentage of Similarity")
ax.set_ylabel("Count of Similarty Values")
plt.hist(tenth_similar_values, bins=30)
plt.title("Tenth Most Similar")

plt.show()
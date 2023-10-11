from hw2 import simValues, simValuesPct
import gensim.downloader as g1

model = g1.load("glove-wiki-gigaword-100")

l1 = simValues(model=model, key="board", countList=[1, 5, 50, 10])

print(l1)

l2 = simValuesPct(model=model, key="board", countPctList=[0, 25, 50, 75, 100])

print(l2)
# download data if doing first run or to update
import nltk
nltk.download('stopwords')
nltk.download('inaugural')
nltk.download('punkt')
from nltk.corpus import inaugural, PlaintextCorpusReader
from CorpusReader_TFIDF import *



print(len(inaugural.words()))
print(inaugural.sents())
print(len(inaugural.sents()))
print(inaugural.fileids())
print(inaugural.sents(['1789-washington.txt']))



myCorpus = CorpusReader_TFIDF(inaugural, tf="log", stopWord="standard")
# TEST: standard set of corpus functions
print(myCorpus.tfidf('1789-Washington.txt'), True)

print("-----\n")

q = myCorpus.tfidfAll()
for x in q:
   print(x, q[x])

print("-----\n")

docs_list = ['1789-Washington.txt', '2021-Biden.txt']
print(myCorpus.cosine_sim(docs_list))

print("-----\n")

print(myCorpus.cosine_sim_new(['citizens', 'economic', 'growth', 'economic'], '2021-Biden.txt'))


#  This is for testing your own corpus
#
#  create a set of text files, store them in a directory specified from 'rootDir' variable
#
#  

rootDir = '/myhomedirectory'   # change that to the directory where the files are
newCorpus = PlaintextCorpusReader(rootDir, '*')
tfidfCorpus = CorpusReader_TFIDF(newCorpus)

q = tfidfCorpus.tfidfAll()
for x in q:
   print(x, q[x])

print("-----\n")

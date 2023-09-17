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

coolidge_nixon_ai_speech = [
    "Ladies", "and", "gentlemen", "fellow", "Americans",
    "I", "stand", "before", "you", "today", "to", "address", "the", "critical", "issues", "of", "our", "time",
    "for", "the", "challenges", "we", "face", "are", "formidable", "but", "together", "as", "a", "united", "nation",
    "we", "shall", "prevail",
    "In", "the", "spirit", "of", "President", "Coolidge's", "simplicity", "and", "clarity", "let", "us", "remember",
    "that", "government's", "role", "should", "be", "limited", "for", "it", "is", "the", "people", "who", "are", "the",
    "architects", "of", "their", "own", "destiny", "We", "must", "respect", "the", "wisdom", "of", "our", "Constitution",
    "which", "provides", "a", "framework", "for", "our", "nation's", "success", "As", "Coolidge", "once", "said",
    "'The", "business", "of", "America", "is", "business'" "and", "indeed", "a", "thriving", "economy", "is", "the",
    "cornerstone", "of", "our", "prosperity",
    "Yet", "as", "President", "Nixon", "reminded", "us", "we", "cannot", "ignore", "the", "world", "beyond", "our",
    "borders", "We", "must", "engage", "in", "diplomacy", "for", "peace", "is", "not", "the", "absence", "of", "conflict",
    "but", "the", "presence", "of", "justice", "In", "the", "pursuit", "of", "peace", "we", "shall", "not", "falter",
    "In", "times", "of", "adversity", "it", "is", "our", "resilience", "that", "defines", "us", "as", "a", "nation", "We",
    "face", "economic", "challenges", "but", "our", "spirit", "of", "innovation", "shall", "guide", "us", "through", "We",
    "face", "global", "tensions", "but", "our", "commitment", "to", "diplomacy", "shall", "be", "unwavering",
    "As", "President", "Coolidge", "championed", "individual", "liberty", "let", "us", "remember", "that", "freedom",
    "is", "the", "bedrock", "of", "our", "society", "But", "as", "President", "Nixon", "understood", "with", "great",
    "freedom", "comes", "great", "responsibility", "We", "must", "ensure", "that", "our", "liberty", "is", "tempered",
    "with", "justice", "for", "true", "greatness", "lies", "in", "our", "moral", "character",
    "In", "closing", "let", "us", "heed", "the", "lessons", "of", "both", "presidents", "Let", "us", "embrace",
    "simplicity", "and", "clarity", "champion", "individual", "liberty", "and", "engage", "with", "the", "world",
    "while", "never", "losing", "sight", "of", "our", "values", "Together", "we", "can", "overcome", "any", "challenge",
    "and", "forge", "a", "brighter", "future", "for", "all", "Americans",
    "Thank", "you", "and", "may", "God", "bless", "the", "United", "States", "of", "America"
]

query_test = myCorpus.query(words=coolidge_nixon_ai_speech)
print(query_test)
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

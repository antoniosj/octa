from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer


def lower_words(words):
    return [word.lower() for word in words]


def remove_punctuation(words):
    return [word for word in words if word.isalnum()]


def remove_stopwords(words):
    sw = stopwords.words('english')
    return [word for word in words if word not in sw]


def stem(words):
    stemmer = EnglishStemmer()
    return [stemmer.stem(word) for word in words]


def preprocess_words(words):
    lowered_words = lower_words(words)
    without_punct = remove_punctuation(lowered_words)
    without_punct_and_stopwords = remove_stopwords(without_punct)
    without_punct_and_stopwords_and_stemmed = stem(without_punct_and_stopwords)
    return without_punct_and_stopwords_and_stemmed


def preprocess_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        tokenized_sentence = preprocess_words(words)
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences

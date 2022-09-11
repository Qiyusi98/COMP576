
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from scipy import spatial
import string
import numpy as np
import math
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def tokenize(document_string):
        """
        ref: https://www.nltk.org/api/nltk.tokenize.html
        """
        return word_tokenize(document_string)

    @staticmethod
    def punctuation_removal(term_sequence):
        return [word for word in term_sequence if not word in string.punctuation]

    @staticmethod
    def stop_word_removal(term_sequence):
        """
        ref: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
        """
        stop_words_set = set(stopwords.words('english'))
        return [word for word in term_sequence if not word in stop_words_set]

    @staticmethod
    def case_folding(term_sequence):
        return [word.casefold() for word in term_sequence]

    @staticmethod
    def lemmatization(term_sequence):
        """
        ref: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
        :return:
        """
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in term_sequence]

    @staticmethod
    def stemming(term_sequence):
        """
        ref: https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
        :return:
        """
        porter = PorterStemmer()
        return [porter.stem(word) for word in term_sequence]

def pre_process_document(document, debug=False):
    print("Before pre processing, document is ", document)
    if debug:
        print("Start tokenize document")
    document = Preprocessing.tokenize(document)
    if debug:
        print("After tokenize, the term frequency of document is ", document)
        print("Start punctuation removal and case folding")
    document = Preprocessing.punctuation_removal(document)
    document = Preprocessing.case_folding(document)
    if debug:
        print("After punctuation removal and case folding, the term frequency of document is ", document)
        print("Start stop word removal")
    document = Preprocessing.stop_word_removal(document)
    if debug:
        print("After stop word removal, the term frequency of document is ", document)
        print("Start lemmatization and stemming")
    document = Preprocessing.stemming(document)
    document = Preprocessing.lemmatization(document)
    if debug:
        print("After lemmatization and stemming, the term frequency of document is ", document)
    print("After pre processing, document is ", document)
    return document

def initialize_matrix(term_document_matrix, document_name_list, term_frequency_list):
    for term_frequency in term_frequency_list:
        for term in term_frequency:
            for document in document_name_list:
                if term not in term_document_matrix:
                    term_document_matrix[term] = {}
                term_document_matrix[term][document] = 0
    return term_document_matrix

def build_term_document_incidence_matrix(term_document_matrix, document_name, term_sequence):
    for term in term_sequence:
        term_document_matrix[term][document_name] += 1
    return term_document_matrix

def build_tf_idf(term_document_matrix, document_list):
    num_of_document = len(document_list)
    idf = {}
    for term in term_document_matrix:
        if term not in idf:
            idf[term] = 0
        for document in term_document_matrix[term]:
            if term_document_matrix[term][document] > 0:
                idf[term] += 1
    for term in idf:
        idf[term] = np.log(((1 + num_of_document) / (1 + np.log(idf[term])))) + 1

    tfidf = term_document_matrix.copy()
    for term in tfidf:
        for document in tfidf[term]:
            tfidf[term][document] *= idf[term]
    for document in document_list:
        square_sum = 0
        for term in tfidf:
            square_sum += tfidf[term][document] ** 2
        for term in tfidf:
            tfidf[term][document] /= math.sqrt(square_sum)
    return tfidf

def get_similarity_between_document(tfidf, document_list):
    similarity = []
    for document_1 in document_list:
        vector_1 = get_document_vector(tfidf, document_1)
        row = []
        for document_2 in document_list:
            vector_2 = get_document_vector(tfidf, document_2)
            cosine = 1 - spatial.distance.cosine(vector_1, vector_2)
            row.append(round(cosine, 4))
        similarity.append(row)
    return similarity

def get_document_vector(tfidf, document):
    vector = []
    for term in tfidf:
        vector.append(tfidf[term][document])
    return vector

def convert_to_matrix(dict):
    list = []
    for term in dict:
        row = []
        for doc in dict[term]:
            row.append(dict[term][doc])
        list.append(row)
    return np.array(list)

if __name__ == '__main__':
    document_1 = "Today DeepMind released AlphaCode!!!!!! Go AlphaCode!"
    document_2 = "The AlphaCode is released today."
    document_3 = "the AlphaCode won last week."
    document_4 = "Find the AlphaCode news last week."
    document_5 = "AlphaCode is a system released by DeepMind."

    document_1 = pre_process_document(document_1)
    document_2 = pre_process_document(document_2)
    document_3 = pre_process_document(document_3)
    document_4 = pre_process_document(document_4)
    document_5 = pre_process_document(document_5)

    term_document_matrix = {}
    document_name_list = ["document_1", "document_2", "document_3", "document_4", "document_5"]
    document_map = {}
    document_map["document_1"] = document_1
    document_map["document_2"] = document_2
    document_map["document_3"] = document_3
    document_map["document_4"] = document_4
    document_map["document_5"] = document_5

    term_frequency_list = list(document_map.values())
    term_document_matrix = initialize_matrix(term_document_matrix, document_name_list, term_frequency_list)
    for document_name in document_name_list:
        term_document_matrix = build_term_document_incidence_matrix(term_document_matrix, document_name, document_map[document_name])
    print("Term document incidence matrix with label is ")
    for term in term_document_matrix:
        print(term, term_document_matrix[term])
        # print(term_document_matrix[term])
    print("Term document incidence matrix is \n", convert_to_matrix(term_document_matrix))

    # vectorizer = TfidfVectorizer(stop_words='english')
    # response = vectorizer.fit_transform([document_1, document_2, document_3, document_4, document_5])
    # df_tfidf_sklearn = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names())
    # print(df_tfidf_sklearn)

    tfidf = build_tf_idf(term_document_matrix, document_name_list)
    print("TF-IDF matrix with label is ")
    for term in tfidf:
        print(term, tfidf[term])
    print("TF-IDF matrix is \n", convert_to_matrix(tfidf))

    similarity = get_similarity_between_document(tfidf, document_name_list)
    matrix = np.array(similarity)
    print("Similarity matrix between document is \n", matrix)



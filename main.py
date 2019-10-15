12#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:17:24 2019

@author: temperantia
refference : https://stackabuse.com/python-for-nlp-creating-tf-idf-model-from-scratch/
"""
import numpy as np
import nltk
from os import listdir
from os.path import isfile, join
from sklearn.metrics.pairwise import cosine_similarity

def getCorpus(folders):
    corpus = []
    corpusNames = []
    for files in folders:
        for data in files[1]:
            file = open(data, "r") 
            data = file.read()
            data = data.splitlines()[4]
            data = data.lower()
            corpus.append(data)
        for name in files[0]:
            corpusNames.append(name)
    return corpus, corpusNames

def returnListOfFilePaths(folderPath):
    fileInfo = []
    listOfFileNames = [fileName for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    listOfFilePaths = [join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    fileInfo.append(listOfFileNames)
    fileInfo.append(listOfFilePaths)
    return fileInfo

def preprocess(datasetSrc):
    folders= [returnListOfFilePaths(folder) for folder in datasetSrc]
    
    corpus, corpusNames = getCorpus(folders)
    #tokenization
    corpus2 = []
    for item in corpus:
        corpus2.append(item.lower())
    corpus = corpus2
    word = ""
    for item in corpus:
        word+=item+" "
    tokenized = nltk.tokenize.word_tokenize(word)
    tokenized = list(set(tokenized))
    #remove stopword
    stop_word_set = set(nltk.corpus.stopwords.words("indonesian"))
    filteredContents = [word for word in tokenized if word not in stop_word_set]
    #stemming
    tokenized = filteredContents
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [porterStemmer.stem(word) for word in tokenized]
    #remove puncuation
    excludePuncuation = [",",'.','{','}','(',')', '-', '!', '+']
    tokenized= [word for word in tokenized if word not in excludePuncuation]
    return tokenized, corpus, corpusNames

def getIDF(tokenized, corpus):
    word_idf_values = {}
    token_each_doc = []
    for document in corpus:
        token_each_doc.append(nltk.word_tokenize(document))
    for token in tokenized:
        doc_containing_word = 0
        for token_doc in token_each_doc:
            if token in token_doc:
                doc_containing_word += 1
        word_idf_values[token] = np.log10(len(corpus)/(1 + doc_containing_word))
    return word_idf_values
    
def getTF(tokenized, corpus):
    word_tf_values = {}
    token_each_doc = []
    for document in corpus:
        token_each_doc.append(nltk.word_tokenize(document))
    for token in tokenized:
        sent_tf_vector = []
        for document in token_each_doc:
            doc_freq = 0
            for word in document:
                if token == word:
                      doc_freq += 1
            """
            if (doc_freq != 0):
                word_tf = 1 + np.log10(doc_freq)
            else:
                word_tf = 0
            """
            word_tf = doc_freq/len(document)
            sent_tf_vector.append(word_tf)
        word_tf_values[token] = sent_tf_vector
    return word_tf_values

def getTFIDF(tokenized, corpus, word_tf_values, word_idf_values):
    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_sentences = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_sentences.append(tf_idf_score)
        tfidf_values.append(tfidf_sentences)
    tfidf = np.transpose(tfidf_values)
    return tfidf

def main():
    datasetSrc = ["../Dataset/Bisnis/",
                  "../Dataset/Edukasi/",
                  "../Dataset/Internasional/",
                  "../Dataset/Metropolitan/",
                  "../Dataset/Nasional/",
                  "../Dataset/Olahraga/",
                 
                  ]
    tokenized, corpus, corpusNames = preprocess(datasetSrc)
    IDF = getIDF(tokenized,corpus)
    TF = getTF(tokenized, corpus)
    TFIDF = getTFIDF(tokenized, corpus, TF, IDF)
    similarity = cosine_similarity (TFIDF)
    
    i=0
    for i in range(3):
        print("dokumen awal :", corpusNames[i])
        distance = similarity[i]
        closest = 0
        closest_place = 0
        results = [i,]
        results_score = [1.0,]
        b = 0
        for b in range (5):
            a=0
            closest = 0
            for a in range (len(distance)):
                if distance[a]>closest and a not in results:
                    closest = distance[a]
                    closest_place = a
            results.append(closest_place)
            results_score.append(closest)
        print ("dokumen hasil :")
        a = 0
        for a in range(len(results)):
            print("judul: ", corpusNames[results[a]], " score: ", results_score[a])
    
    
    
    
main()



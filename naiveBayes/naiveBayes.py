# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:01:12 2023

@author: ab3935
"""
import csv
import random

def ParseFeatures(file_path):
    features = {}
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            features[row[0]] = Article(0 if row[2] == "objective" else 1)
    return features

def GetWordsFromArticle(path):
    with open(path, encoding="ISO-8859-1") as f:
        words = []
        for row in f.read().split():
            # Remove all special characters
            word = ''.join(e for e in row if e.isalnum())
            words.append(word)
    return words

class Article:
    def __init__(self, label):
         self.label = label
         self.words = []

def BagOfWords(wordarr):
    wordCount = {}
    
    for word in wordarr:
            if wordCount.get(word) != None:
                wordCount[word] += 1
            else:
                wordCount[word] = 1
    return wordCount

class NaiveBayes:
    wordCounts = {}
    def __init__(self):
        self.wordCounts = {}

    def add_to_word_count(self,text,label):
        listOfWords = BagOfWords(text)
        if self.wordCounts.get(label) == None:
            self.wordCounts[label] = listOfWords
        else:
            for word in listOfWords:
                if self.wordCounts[label].get(word) == None:
                    self.wordCounts[label][word] = listOfWords[word]
                else:
                    self.wordCounts[label][word] += listOfWords[word]
        return
    
    def add_word_threshold(self, threshold):
        items_below_threshold = []
        for label in self.wordCounts.keys():
            for word in self.wordCounts[label]:
                if self.wordCounts[label][word] <= threshold:
                     items_below_threshold.append([label, word])
        
                     
        for item in items_below_threshold:    
            del self.wordCounts[item[0]][item[1]]

    def predict_text(self,text):
        wordarr = BagOfWords(text)
        labels = list(self.wordCounts.keys())
        probabilities = {}
        
        for label in labels:
            probabilities[label] = 1
            
        #find highest probility
        for word in wordarr:
            for label in labels:
                prob = self.calculate_probability(word,label)
                if prob > 0:
                    probabilities[label] *= prob
        
        highest = probabilities[labels[0]]
        highestlabel = labels[0]
        for label in labels:
            #print(probabilities[label])
            if probabilities[label] > highest:
                #print("prev highest", highestlabel,'new:',label)
                highest = probabilities[label]
                highestlabel = label
     
        return highestlabel
        
    
    def calculate_probability(self,word,lab):
        labels = self.wordCounts.keys()
        total = 0
        labcount = self.wordCounts[lab].get(word,0)
        for label in labels:
            total += self.wordCounts[label].get(word,0)
        #print(word," ",lab,":",labcount,total,labcount/total )
        if (labcount == 0 or total == 0):
            return 0
        else:
            return labcount/total

class CrossValidationManager:
    k = 0
    data_length = 0
    data_indexes = {}
    current_index = -1

    def __init__(self, k_times, data, seed = 0):
        self.k = k_times
        self.data = data
        if (seed > 0):
            random.Random(seed).shuffle(self.data)
        else:
            random.shuffle(self.data)
        self._get_indexes()
    
    def _get_indexes(self):
        length = int(len(self.data) / self.k)
        remainder = len(self.data) % self.k

        if remainder > 0:
            length += 1
        else:
            remainder = -1

        for index in range(self.k):
            if remainder > 0:
                remainder -= 1
            elif remainder == 0:
                length -= 1
                remainder = -1

            if index == 0:
                self.data_indexes[index] = [0,length]
            else:
                previous_end_index = self.data_indexes[index-1][1]
                self.data_indexes[index] = [previous_end_index, previous_end_index + length]
    
    def data_available(self):
        self.current_index += 1
        if self.current_index == self.k:
            return False
        return True

    def get_training_data(self):
        data = []
        for i in range(self.k):
            if i == self.current_index:
                continue
            s = self.data_indexes[i][0]
            e = self.data_indexes[i][1]
            data += self.data[s:e]
        return data
    
    def get_validation_data(self):
        s = self.data_indexes[self.current_index][0]
        e = self.data_indexes[self.current_index][1]
        return self.data[s:e]

# Data: https://archive.ics.uci.edu/ml/machine-learning-databases/00450/
# Label: 0 = Objective, 1 = Subjective


K_FOLD_COUNT = 5
# Changing the seed will shuffle the data differently but consistently between runs
# 0 will shuffle differently every run of the program
SEED = 12


data = ParseFeatures("Data/features.csv")
for key in data:
    data[key].words = GetWordsFromArticle("Data/" + key + ".txt")
 
dataManager = CrossValidationManager(K_FOLD_COUNT, list(data), seed=SEED)

while dataManager.data_available():
    n = NaiveBayes()
    trainData = dataManager.get_training_data()
    valData = dataManager.get_validation_data()

    # Train
    for article_name in trainData:
        n.add_to_word_count(GetWordsFromArticle("Data/" + article_name + ".txt"), data[article_name].label)
    
    #n.add_word_threshold(2)

    #Test
    validationData = {}
    for article_name in valData:
        validationData[article_name] = [GetWordsFromArticle("Data/" + article_name + ".txt"), data[article_name].label]

    correct = 0
    for words in validationData.items():
        prediction = n.predict_text(words[1][0])
        
        if prediction == words[1][1]:
            correct += 1
    print((correct / len(validationData)) * 100, end='')
    print(" Correct: " + str(correct) + "/" + str(len(validationData)))

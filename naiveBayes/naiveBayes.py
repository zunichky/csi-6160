import os
import re
import csv
import numpy as np

# Data: https://archive.ics.uci.edu/ml/machine-learning-databases/00450/
# Label: 0 = Objective, 1 = Subjective

def parse_features(file_path):
    features = {}
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            features[row[0]] = article(0 if row[2] == "objective" else 1)

    return features

def parse_articles(path):
    with open(path, encoding="ISO-8859-1") as f:
        words = {}
        for row in f.read().split():
            # Remove all special characters
            word = ''.join(e for e in row if e.isalnum())
            if word in words.keys():
                words[word] += 1
            else:
               words[word] = 1 
    return words

def merge_totals(input, total):
    for key in input:
        if key in total.keys():
            total[key] = total[key] + input[key]
        else:
            total[key] = input[key] 
    return total

def conditional_probability(words, total):
    cp = {}
    for key in words:
        cp[key] = words[key] / total
    return cp

class article:
    def __init__(self, label):
         self.label = label
         self.words = {}
         self.word_count = -1


#data_set = parse_features("Data/features.cscv")
DATAPATH = "Data"

data = parse_features(os.path.join(DATAPATH,"features.csv"))

for key in data:
    data[key].words = parse_articles(os.path.join(DATAPATH, key + ".txt"))
    #print(key + ": " + str(len(data[key].words)))

# We now have all the data
# Label: 0 = Objective, 1 = Subjective

# Get count of words for each label
objective_words = {}
subjective_words = {}


for key in data:
    count = 0
    if data[key].label == 0:
        objective_words = merge_totals( data[key].words, objective_words )
    else:
        subjective_words = merge_totals( data[key].words, subjective_words )


objective_cp  = conditional_probability(objective_words, sum(objective_words.values()) )
subjective_cp = conditional_probability(subjective_words, sum(subjective_words.values()) )
#print({k: v for k, v in sorted(objective_cp.items(), key=lambda item: item[1])})

'''
 objective_cp contains dict of each words probability 
 Ex) { 'to': 0.026924490143420035, 'the': 0.05395951020310619 }

  P6.pdf, slide 56/57
 Implement prediction
 Ex) input: ["firstround", "or", "players"]
 Calc prob
 obj_prob = objective_cp["firstround"] * objective_cp["or"] * objective_cp["players"]
 sub_prob = subjective_cp["firstround"] * subjective_cp["or"] * subjective_cp["players"]
 if obj_prob > sub_prob -> predict 0, else predict 1


 How do we handle if both predictions have a word that has a 0 probability (slide 58)? Assume we just ignore
 '''
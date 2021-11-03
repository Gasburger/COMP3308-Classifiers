import sys
from csv import reader, writer
from collections import Counter
import numpy as np
import math
import random

# reading csv file and parsing string arguments (excl. class variable) as floats
def load_train(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    for i in range(len(dataset[0])-1):
        for row in dataset:
            row[i] = float(row[i].strip())

    return dataset

def load_test(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    for i in range(len(dataset[0])):
        for row in dataset:
            row[i] = float(row[i].strip())

    return dataset

def knn(train_set, test_set, k):
    distance = []
    for group in train_set:
        for features in train_set[group]:
            dist_euc = np.sqrt(np.sum((np.array(features) - np.array(test_set)) ** 2))
            distance.append([dist_euc, group]) 
    
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

def knn_predict(train_data, test_data, k):
    vote_collector = []
    train_set = {'yes':[], 'no': []}
    test_set = []

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set.append(i)

    for data in test_set:
        vote = knn(train_set, data, k)
        vote_collector.append(vote)

    return vote_collector
    #print("Accuracy:", correct/t)

def nbayes(train_data, test_data):
    train_set = {'yes':[], 'no': []}
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    
    # 2d numpy mean/standard deviation code used from 
    # https://stackoverflow.com/questions/15819980/calculate-mean-across-dimension-in-a-2d-array
    mean_yes = list(np.mean(train_set['yes'], axis=0))
    mean_no = list(np.mean(train_set['no'], axis=0))
    sd_yes = list(np.std(train_set['yes'], axis=0))
    sd_no = list(np.std(train_set['no'], axis=0))

    pdf_yes = []
    pdf_no = []
    
    for x in test_data:
        for i in range(len(mean_yes)):
            pdf_yes.append((1.0 / (sd_yes[i] * math.sqrt(2*math.pi))) * math.exp(-0.5*((x[i] - mean_yes[i]) / sd_yes[i]) ** 2))
    
    for x in test_data:
        for i in range(len(mean_no)):
            pdf_no.append((1.0 / (sd_no[i] * math.sqrt(2*math.pi))) * math.exp(-0.5*((x[i] - mean_no[i]) / sd_no[i]) ** 2))

    #pdf_list = [(pdf_yes[i], pdf_no[i]) for i in range(0, len(pdf_yes))]
    #pdf_chunks = [pdf_list[x:x+len(mean_yes)] for x in range(0, len(pdf_list), len(mean_yes))]
    pdf_yes = [pdf_yes[x:x+len(mean_yes)] for x in range(0, len(pdf_yes), len(mean_yes))]
    pdf_no = [pdf_no[x:x+len(mean_no)] for x in range(0, len(pdf_no), len(mean_no))]
    
    val_yes = []
    val_no = []

    for x in pdf_yes:
        val_yes.append((268/768) * np.prod(x))

    for y in pdf_no:
        val_no.append((500/768) * np.prod(y))
    
    votes = []
    for i in range(len(pdf_yes)):
        if val_yes[i] > val_no[i]:
            votes.append('yes')
        elif val_yes[i] < val_no[i]:
            votes.append('no')
        else:
            votes.append('yes')

    return votes

def main(argv):
    trainf = sys.argv[1]
    testf = sys.argv[2]
    classifier = sys.argv[3]

    train_data = load_train(trainf)
    test_data = load_test(testf)

    if classifier == "NB":
        val = nbayes(train_data, test_data)
        for v in val:
            print(v)
    elif "NN" in classifier:
        k = int(classifier.strip("NN"))
        votes = knn_predict(train_data, test_data, k)
        for v in votes:
            print(v)

if __name__ == "__main__":
   main(sys.argv)
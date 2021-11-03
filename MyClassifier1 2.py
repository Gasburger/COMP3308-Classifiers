import csv
from collections import Counter
import numpy as np
import random

# reading csv file and parsing string arguments (excl. class variable) as floats
pima = []
test = []
with open('pima.csv', 'r') as file:
    pima_reader = csv.reader(file, delimiter=',')
    for row in pima_reader:
        if not row:
            continue
        pima.append(row)

with open('test.csv', 'r') as file:
    test_reader = csv.reader(file, delimiter=',')
    for row in test_reader:
        test.append(row)

for i in range(len(pima[0])-1):
    for row in pima:
        row[i] = float(row[i].strip())

for i in range(len(test[0])-1):
    for row in test:
        row[i] = float(row[i].strip())

k_val = 5 
def knn(dataset, prediction, k=k_val):
    distance = []
    for group in dataset:
        for features in dataset[group]:
            dist_euc = np.sqrt(np.sum((np.array(features) - np.array(prediction)) ** 2))
            distance.append([dist_euc, group]) 
    
    votes = [i[1] for i in sorted(distance)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

#test_size = 0.2
train_set = {'yes':[], 'no': []}
test_set = []
train_data = pima
test_data = test

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set.append(i)

correct = 0
t = 0

for data in test_set:
    print(data)
    #vote = knn(train_set, data, k=k_val)
    #print(vote)

#print("Accuracy:", correct/t)





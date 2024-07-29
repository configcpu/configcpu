# Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file.

import csv
num_attribute = 6
a = []
with open('enjoysport.csv', 'r') as file:
    reader = csv.reader(file)
    a = list(reader)

hypothesis = a[0][:-1]

for i in a:
    if i[-1] == 'yes':
        for j in range(num_attribute):
            if i[j] != hypothesis[j]:
                hypothesis[j] = '?'
print(hypothesis)
print("\nThe Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)

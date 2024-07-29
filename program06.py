# For a given set of training data examples stored in a .CSV file, implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.
import csv

with open("program06.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)
    
    s = data[1][:-1]  
    g = [["?" for _ in range(len(s))] for _ in range(len(s))]  

    for example in data[1:]:  
        attributes = example[:-1]  
        label = example[-1]  
        
        if label.lower() == "yes":
            for j in range(len(s)):
                if attributes[j] != s[j]:
                    s[j] = "?"
                    g[j][j] = "?"
        elif label.lower() == "no":
            for j in range(len(s)):
                if attributes[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"

gh = [h for h in g if any(val != "?" for val in h)]

print("\nSteps of Candidate Elimination Algorithm", len(data) - 1)
print(s)
print(g)
print("\nFinal specific hypothesis:\n", s)
print("\nFinal general hypothesis:\n", gh)


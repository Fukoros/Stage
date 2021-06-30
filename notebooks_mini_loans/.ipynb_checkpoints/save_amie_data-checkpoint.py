import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def retrieve_indexes(root, cv):
    indexes = []

    for i in range(cv):
        f = open(f"{root}index_{i}.tsv", "r")

        lines = f.readlines()
        train_index = np.array(list(map(int, lines[0].split(","))))
        test_index = np.array(list(map(int, lines[1].split(","))))
        indexes.append((train_index, test_index))
        f.close()
        
    return indexes

# Transform an integer id to a string id
def int_to_str_id(idsInt):
    res = chr(97+idsInt%26)
    idsReducted = int(idsInt/26)
    while idsReducted != 0:
        res = chr(97+(idsReducted-1)%26) + res
        idsReducted = int(idsReducted/26)
    return res

# Syntax of Amie data
def syntax(subject, predicate, objectD, integer=True):
    if integer:
        return f"{subject}\t{predicate}\t{objectD} \n" # Integer id
    else:
        return f"{int_to_str_id(subject)}\t{predicate}\t{objectD} \n" #String id

# Create the text for each feature and finally write it to the file
def formatData(f, idData, data, integer=True):
    toWrite = ""
    for i in data.index:
        toWrite += syntax(idData, i, data.loc[i], integer=integer)
    f.write(toWrite)

def save_all_data(root, booleanDF, integer=True):
    f = open(root+"Knowledge_Data.tsv", "w")

    booleanDF_to_Save = booleanDF

    for idData in booleanDF_to_Save.index:
        formatData(f, idData, booleanDF_to_Save.iloc[idData], integer=integer)

    f.close()
    
def save_CV(root, booleanDF, cv, integer=True):
    booleanDF_to_Save = booleanDF

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    kf.get_n_splits(booleanDF_to_Save)

    cp = 0

    for train_index, test_index in kf.split(booleanDF_to_Save):

        f = open(root+f"CV_train_{cp}.tsv", "w")

        for idData in train_index:
            formatData(f, idData, booleanDF_to_Save.iloc[idData], integer=integer)

        f.close()

        print("Train repartition :\n",booleanDF_to_Save.iloc[train_index]["approval"].value_counts())

        f = open(root+f"CV_valid_{cp}.tsv", "w")

        for idData in test_index:
            formatData(f, idData, booleanDF_to_Save.iloc[idData], integer=integer)

        f.close()

        f = open(root+f"index_{cp}.tsv", "w")

        f.write(str(train_index[0]))
        for i in range(1,len(train_index)):
            f.write(","+str(train_index[i]))
        f.write("\n")
        f.write(str(test_index[0]))
        for i in range(1,len(test_index)):
            f.write(","+str(test_index[i]))

        f.close()

        cp += 1
import numpy as np

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
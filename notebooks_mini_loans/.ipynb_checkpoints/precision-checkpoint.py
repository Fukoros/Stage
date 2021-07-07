import pandas as pd
import numpy as np
from multiprocessing import Process, Manager, SimpleQueue
import multiprocessing

def prediction_right(X, variables, rule):
    map_variable = {}
    for i, variable in enumerate(variables): 
        #If we have a variable
        if variable:
            if not rule.hypotheses[i].objectD in map_variable.keys():
                map_variable[rule.hypotheses[i].objectD] = str(X.iloc[i])
            elif map_variable[rule.hypotheses[i].objectD] != str(X.iloc[i]):
                return "Can't say anything"
        #If we have a Const
        else:
            if rule.hypotheses[i].objectD != str(X.iloc[i]):
                return "Can't say anything"
            
    if not (rule.conclusion.objectD[0] != "?"):
        return str(map_variable[rule.conclusion.objectD] == str(X["approval"]))
    else:
        return str(rule.conclusion.objectD == str(X["approval"]))
    
def compute_precisions(rules, df, rules_result, index, cptShared, train_index, test_index):
    print(f"Process n°{index} : Launched")
    
    for rule in rules:  
        columns = [rule.hypotheses[k].predicate for k in range(len(rule.hypotheses))]
        columns.append("approval")

        res = df[columns].loc[train_index].apply(func=prediction_right, axis=1, variables=[not (hypothese.objectD[0] != "?") for hypothese in rule.hypotheses],\
                                                 rule=rule).value_counts()

        if not "False" in res.index:
            res["False"] = 0

        if not "True" in res.index:
            res["True"] = 0

        rule.setPrecisionTrain(res["True"] / (res["True"]+res["False"]))

        res = df[columns].loc[test_index].apply(func=prediction_right, axis=1, variables=[not (hypothese.objectD[0] != "?") for hypothese in rule.hypotheses],\
                                                 rule=rule).value_counts()

        if not "False" in res.index:
            res["False"] = 0

        if not "True" in res.index:
            res["True"] = 0

        if np.isnan(res["True"] / (res["True"]+res["False"])):
            print(res, rule)
        rule.setPrecisionTest(res["True"] / (res["True"]+res["False"]))

        rules_result[str(rule)] = rule

        cptShared.value += 1
        if (cptShared.value%100 == 0):
            print(cptShared)
        
    print(f"Process n°{index} : Finished")
    
def run_precision(root, indexes, rules_per_cv):
    rules_per_CV = {}

    df = pd.read_csv(root+"dfSave.csv", index_col=0)

    for para in rules_per_cv:
        with Manager() as manager:
            rules = rules_per_cv[para]

            train_index, test_index = indexes[int(para.split("-")[0].split("=")[1])]

            rules_result =  manager.dict()

            cpt_total = manager.Value("d",0)

            processes_to_create = multiprocessing.cpu_count()-3
            processes = list()

            rules_list = list(rules)

            for index in range(processes_to_create):
                x = Process(target=compute_precisions, args=(rules_list[int(np.floor(index*len(rules_list)/processes_to_create)): int(np.floor((index+1)*len(rules_list)/processes_to_create))], df, rules_result, index, cpt_total, train_index, test_index))
                processes.append(x)
                x.start()

            for index, process in enumerate(processes):
                process.join()

            rules_per_CV[para] = rules_result.copy()
    return rules_per_CV

# Given different folds of CV per parameters, it will return the mean and std for the precisions of each rules on the train and test set
def mean_and_std_precision(dfs, cv):
    already_seen = []
    to_return = {}
    to_return_std = {}
    for key in dfs.keys():
        key = key.split("-")
        if not (key[1] in already_seen):
            precision_train = []
            precision_test = []            
            for i in range(cv):
                df = dfs["CV="+str(i)+"-"+key[1]]
                precision_train.append(df["precision_train"].mean())
                precision_test.append(df["precision_test"].mean())
            to_return[key[1]] = {"precision_train":np.mean(precision_train), "precision_test":np.mean(precision_test)}
            to_return_std[key[1]] = {"precision_train":np.std(precision_test), "precision_test":np.std(precision_train)}
            
            already_seen.append(key[1])
    return pd.DataFrame.from_dict(to_return, orient="index"), pd.DataFrame.from_dict(to_return_std, orient="index")
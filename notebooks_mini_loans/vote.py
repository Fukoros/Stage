import pandas as pd
import numpy as np
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os.path
from os import path
from sklearn.model_selection import KFold
from multiprocessing import Process, Manager, SimpleQueue
import multiprocessing
import shutil
from rules import *
from vote import *

root = "./../data_cv/"
cv = 3

# ---------------------------------------- SYSTEM OF VOTE -----------------------------------

def safe(X, rules_to_predict):
    predicted_approved = False
    for rule in rules_to_predict.values:
        variables=[not (hypothese.objectD == "False" or hypothese.objectD == "True") for hypothese in rule[0]]

        map_variable = {}
        rule_can_apply = True
        for i, variable in enumerate(variables): 
            #If we have ?b 
            if variable:
                if not rule[0][i].objectD in map_variable.keys():
                    map_variable[rule[0][i].objectD] = str(X.iloc[i])
                elif map_variable[rule[0][i].objectD] != str(X.iloc[i]):
                    rule_can_apply = False
            #If we have a True or False
            else:
                if rule[0][i].objectD != str(X.iloc[i]):
                    rule_can_apply = False
        if rule_can_apply:
            if not (rule[1].objectD == "False" or rule[1].objectD == "True"):
                if map_variable[rule[1].objectD] == "False":
                    return "Not Approved"
            else:
                if rule[1].objectD == "False":
                    return "Not Approved"
        
    if predicted_approved:
        return "Approved"
    else:
        return "Not Approved - No rules was able to say anything"
    

def democracy(predicts):
    if not "False" in predicts:
        if not "True" in predicts:
            return "Not Approved - No rules was able to say anything"
        else:
            return "Approved"
    elif not "True" in predicts:
        return "Not Approved"
    else:
        if predicts["True"] < predicts["False"]:
            return "Not Approved"
        else:
            return "Approved"
        
def democracy_proportional(predicts, proportion):
    if not "False" in predicts:
        if not "True" in predicts:
            return "Not Approved - No rules was able to say anything"
        else:
            return "Approved"
    elif not "True" in predicts:
        return "Not Approved"
    else:
        if predicts["True"]*proportion["True"] < predicts["False"]*proportion["False"]:
            return "Not Approved"
        else:
            return "Approved"

def expert(X, rules_to_predict):
    #Rules have to be sorted
    
    cpt = 0
    while cpt < len(rules_to_predict):
        
        rule = rules_to_predict.iloc[cpt]        
        variables=[not (hypothese.objectD == "False" or hypothese.objectD == "True") for hypothese in rule[0]]

        map_variable = {}
        rule_can_apply = True
        for i, variable in enumerate(variables): 
            #If we have ?b 
            if variable:
                if not rule[0][i].objectD in map_variable.keys():
                    map_variable[rule[0][i].objectD] = str(X.iloc[i])
                elif map_variable[rule[0][i].objectD] != str(X.iloc[i]):
                    rule_can_apply = False
            #If we have a True or False
            else:
                if rule[0][i].objectD != str(X.iloc[i]):
                    rule_can_apply = False
        if rule_can_apply:
            if not (rule[1].objectD == "False" or rule[1].objectD == "True"):
                if map_variable[rule[1].objectD] == "True":
                    return "Approved"
                else:
                    return "Not Approved"
            else:
                if rule[1].objectD == "True":
                    return "Approved"
                else:
                    return "Not Approved"
        
        cpt+=1
    return "Not Approved - No rules was able to say anything"

# ---------------------------------------- Prepare certains VOTE -----------------------------------

def prepare_vote_democracy_proportional(X, rules_to_predict, proportion):
    prediction = []
    for rule in rules_to_predict.values:
        
        variables=[not (hypothese.objectD == "False" or hypothese.objectD == "True") for hypothese in rule[0]]

        map_variable = {}
        rule_can_apply = True
        for i, variable in enumerate(variables): 
            #If we have ?b 
            if variable:
                if not rule[0][i].objectD in map_variable.keys():
                    map_variable[rule[0][i].objectD] = str(X.iloc[i])
                elif map_variable[rule[0][i].objectD] != str(X.iloc[i]):
                    rule_can_apply = False
            #If we have a True or False
            else:
                if rule[0][i].objectD != str(X.iloc[i]):
                    rule_can_apply = False
        if rule_can_apply:
            if not (rule[1].objectD == "False" or rule[1].objectD == "True"):
                prediction.append(map_variable[rule[1].objectD])
            else:
                prediction.append(rule[1].objectD)
        else:
            prediction.append("I can't say anything")
    predicts = pd.Series(prediction).value_counts()    
    return democracy(predicts, proportion)

def prepare_vote_democracy(X, rules_to_predict):
    prediction = []
    for rule in rules_to_predict.values:
        variables=[not (hypothese.objectD == "False" or hypothese.objectD == "True") for hypothese in rule[0]]

        map_variable = {}
        rule_can_apply = True
        for i, variable in enumerate(variables): 
            #If we have ?b 
            if variable:
                if not rule[0][i].objectD in map_variable.keys():
                    map_variable[rule[0][i].objectD] = str(X.iloc[i])
                elif map_variable[rule[0][i].objectD] != str(X.iloc[i]):
                    rule_can_apply = False
            #If we have a True or False
            else:
                if rule[0][i].objectD != str(X.iloc[i]):
                    rule_can_apply = False
        if rule_can_apply:
            if not (rule[1].objectD == "False" or rule[1].objectD == "True"):
                prediction.append(map_variable[rule[1].objectD])
            else:
                prediction.append(rule[1].objectD)
        else:
            prediction.append("I can't say anything")
    predicts = pd.Series(prediction).value_counts()    
    return democracy(predicts)

# ---------------------------------------- Threads function -----------------------------------

def paralel_prediction_res(queue, name, indexes, new_dfs, df, res_count, res_raw, cpt, vote):
    print(f"Process n°{name} : Launched")
    while not queue.empty():
        para = queue.get()
        rules_to_predict = new_dfs[para]
        
        true = 0
        false = 0
        
        train_index, test_index = indexes[int(para.split("-")[0].split("=")[1])]
        
        if vote == democracy_proportional:
            for rule in rules_to_predict.values:
                if rule[1].predicate ==  "approval-True":
                    true += 1
                else:
                    false += 1

            proportion = {"True":1-true/(true+false+1), "False":1-false/(true+false+1)}
        
            final_prediction_train = df.loc[train_index].apply(prepare_vote_democracy_proportional, rules_to_predict=rules_to_predict, axis=1, proportion=proportion)
            final_prediction_test = df.loc[test_index].apply(prepare_vote_democracy_proportional, rules_to_predict=rules_to_predict, axis=1, proportion=proportion)
        elif vote == democracy_proportional:
        
            final_prediction_train = df.loc[train_index].apply(prepare_vote_democracy, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(prepare_vote_democracy, rules_to_predict=rules_to_predict, axis=1)
            
        elif vote == expert:
            
            rules_to_predict = new_dfs[para].sort_values(["precision_train", "precision_test"], ascending=False)
        
            final_prediction_train = df.loc[train_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            
        else:        
            final_prediction_train = df.loc[train_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
        
        cpt.value += 1
        if (cpt.value%10 == 0):
            print(cpt)
            
        res_count[para] = {"train":pd.Series(final_prediction_train).value_counts(), "test":pd.Series(final_prediction_test).value_counts()}
        res_raw[para] = {"train":pd.Series(final_prediction_train).map({"Not Approved - No rules was able to say anything":False, "Not Approved":False, "Approved":True}), \
                         "test":pd.Series(final_prediction_test).map({"Not Approved - No rules was able to say anything":False, "Not Approved":False, "Approved":True})}
        
    print(f"Process n°{name} : Finished")
    
def paralel_prediction_res_baseline(queue, name, indexes, new_dfs, df, res_count, res_raw, cpt, vote, ranking):
    print(f"Process n°{name} : Launched")
    while not queue.empty():
        para = queue.get()
        rules_to_predict = new_dfs[para]
        
        true = 0
        false = 0
        
        train_index, test_index = indexes[int(para.split("-")[0].split("=")[1])]
        
        if vote == democracy_proportional:
            for rule in rules_to_predict.values:
                if rule[1].predicate ==  "approval-True":
                    true += 1
                else:
                    false += 1

            proportion = {"True":1-true/(true+false+1), "False":1-false/(true+false+1)}
        
            final_prediction_train = df.loc[train_index].apply(prepare_vote_democracy_proportional, rules_to_predict=rules_to_predict, axis=1, proportion=proportion)
            final_prediction_test = df.loc[test_index].apply(prepare_vote_democracy_proportional, rules_to_predict=rules_to_predict, axis=1, proportion=proportion)
        elif vote == democracy:
        
            final_prediction_train = df.loc[train_index].apply(prepare_vote_democracy, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(prepare_vote_democracy, rules_to_predict=rules_to_predict, axis=1)
            
        elif vote == expert:
            rules_to_predict = new_dfs[para].sort_values(ranking[0], ascending=ranking[1])
                
            final_prediction_train = df.loc[train_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            
        else:        
            final_prediction_train = df.loc[train_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
            final_prediction_test = df.loc[test_index].apply(vote, rules_to_predict=rules_to_predict, axis=1)
        
        cpt.value += 1
        if (cpt.value%10 == 0):
            print(cpt)
            
        res_count[para] = {"train":pd.Series(final_prediction_train).value_counts(), "test":pd.Series(final_prediction_test).value_counts()}
        res_raw[para] = {"train":pd.Series(final_prediction_train).map({"Not Approved - No rules was able to say anything":None, "Not Approved":False, "Approved":True}), \
                         "test":pd.Series(final_prediction_test).map({"Not Approved - No rules was able to say anything":None, "Not Approved":False, "Approved":True})}
        
    print(f"Process n°{name} : Finished")

# ---------------------------------------- Manage the threads -----------------------------------
    
def prediction_test(root, cv, new_dfs, vote, indexes, baseline, ranking=None):
    
    prediction_per_rules_count = {}
    prediction_per_rules_raw = {}

    df = pd.read_csv(root+"dfSave.csv", index_col=0)

    q = SimpleQueue()

    for r in list(new_dfs.keys()):
        q.put(r)

    with Manager() as manager:

        processes_to_create = multiprocessing.cpu_count()-3
        processes = list()

        res_count = manager.dict()
        res_raw = manager.dict()
        cpt = manager.Value("d",0)

        for name in range(processes_to_create):
            if baseline: 
                x = Process(target=paralel_prediction_res_baseline, args=(q, name, indexes, new_dfs, df, res_count, res_raw, cpt, vote, ranking))
                processes.append(x)
                x.start()
            else:    
                x = Process(target=paralel_prediction_res, args=(q, name, indexes, new_dfs, df, res_count, res_raw, cpt, vote))
                processes.append(x)
                x.start()

        for index, process in enumerate(processes):
            process.join()

        print(len(res_count))
        prediction_per_rules_count = res_count.copy()
        prediction_per_rules_raw = res_raw.copy()
    return prediction_per_rules_count, prediction_per_rules_raw
    
# ---------------------------------------- Post processing -----------------------------------
    
def compare(prediction, ground_truth):
    if len(prediction) != len(ground_truth):
        print("Different size")
        return None
    return sum(np.array(prediction) == np.array(ground_truth))/float(len(ground_truth))

def mean_and_std_vote(dfs, name) -> pd.DataFrame:
    already_seen = []
    to_return = {}
    to_return_std = {}
    for key in dfs.keys():
        key = key.split("-")
        if not (key[1] in already_seen):
            train = []
            test = []            
            for i in range(cv):
                df = dfs["CV="+str(i)+"-"+key[1]]
                train.append(df["train"])
                test.append(df["test"])
            to_return[key[1]] = {"Train-"+name:np.mean(train), "Test-"+name:np.mean(test)}
            to_return_std[key[1]] = {"Train-"+name:np.std(test), "Test-"+name:np.std(train)}
            
            already_seen.append(key[1])
    return pd.DataFrame.from_dict(to_return, orient="index"), pd.DataFrame.from_dict(to_return_std, orient="index")

def count(dictionary):
    res = 0
    if "Not Approved" in dictionary.keys():
        res += dictionary["Not Approved"]
    if "Approved" in dictionary.keys():
        res += dictionary["Approved"]
    if "Not Approved - No rules was able to say anything" in dictionary.keys():
        return res/(res+dictionary["Not Approved - No rules was able to say anything"])
    else :
        return res/res

def maxScore(X):
    return pd.Series((count(X["train"]), count(X["test"])), index=['train', 'test'])

def bestScorePossible(df, name="Z-Max"):
    return mean_and_std_vote(pd.DataFrame.from_dict(df, orient="index").apply(maxScore, axis=1, result_type="expand").to_dict("index"), name)[0].sort_index()

def merge(df1, df2, df1_std, df2_std):
    return df1.merge(df2, left_index=True, right_index=True), df1_std.merge(df2_std, left_index=True, right_index=True)

def print_accuracy(to_print, to_print_std, max_possible, colors, ylim=[-0.02,1.1]):
    fig, _axs = plt.subplots(nrows=1, ncols=1, figsize=(30,10)) 
    
    half_to_print = int(len(to_print.columns)/2)
    half_colors = int(len(colors)/2)
    
    to_print.sort_index().reindex(sorted(to_print.columns, reverse=True), axis=1).plot(kind="bar", ax=_axs, rot=0, yerr=to_print_std, color=colors[0:half_to_print]+colors[half_colors:half_colors+half_to_print], ylim=ylim)

    axes2 = plt.twinx()
    for i in range(len(to_print)):
        axes2.plot([-0.25+i, 0+i], [max_possible.iloc[i]["Train-Z-Max"], max_possible.iloc[i]["Train-Z-Max"]], color="black")
        axes2.plot([0+i, 0.25+i], [max_possible.iloc[i]["Test-Z-Max"], max_possible.iloc[i]["Test-Z-Max"]], color="red")
    axes2.set_ylim(ymin=ylim[0], ymax=ylim[1])
    
def compute_accuracy_prediction(prediction_per_rules_raw, df, indexes):
    accuracy_prediction = {}
    
    for key in prediction_per_rules_raw:
        train_index, test_index = indexes[int(key.split("-")[0].split("=")[1])]
        accuracy_prediction[key] = {"train":compare(prediction_per_rules_raw[key]["train"], (df.loc[train_index])["approval"].values)}
        accuracy_prediction[key]["test"] = compare(prediction_per_rules_raw[key]["test"], (df.loc[test_index])["approval"].values)
        
    return accuracy_prediction
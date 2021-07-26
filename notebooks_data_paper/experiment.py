import pandas as pd

def create_query(rule, tail):
    knows_query = """
    SELECT DISTINCT ?a
    WHERE {"""
    for hyp in rule.hypotheses:
        knows_query+="\n"
        if hyp.subject == "?b":
            knows_query+=f"{tail} "
        else:
            knows_query+=f"{hyp.subject} "
        knows_query+=f"{hyp.predicate} "
        if hyp.objectD == "?b":
            knows_query+=f"{tail} ."
        else:
            knows_query+=f"{hyp.objectD} ."
            
#     knows_query+="\n"+rule.conclusion.subject+" "+rule.conclusion.predicate+" "+tail+" ."
    return knows_query+"""\n}"""

def democracy(pred_sets):
    vote = {}
    
    for pred_set in pred_sets["Prediction"]:
        for pred in pred_set:
            if pred in vote:
                vote[pred] += 1
            else:
                vote[pred] = 1
    
    return pd.DataFrame.from_dict(vote, orient="index", columns=["Vote"]).sort_values(by="Vote", ascending=False)
    
    

def expert(rules):
    rules = rules.sort_values(by="Pca Confidence", ascending=True)
    if len(rules) != 0:
        best_pred = rules.iloc[0]["Prediction"]
    
        vote = {}
        for pred in best_pred:
            vote[pred] = 0

        for pred_set in rules["Prediction"]:
            for pred in pred_set:
                if pred in vote:
                    vote[pred] += 1

        return pd.DataFrame.from_dict(vote, orient="index", columns=["Vote"]).sort_values(by="Vote", ascending=False)
    else:
        return pd.DataFrame.from_dict({}, orient="index", columns=["Vote"]).sort_values(by="Vote", ascending=False)
    
def hit_at(df_prediction, vote, number=10):
    if number <= 0:
        return -1
    
    good_pred = 0
    bad_pred = 0

    for df_pred in df_prediction:
        if df_pred[0][1:-1] in vote(df_prediction[df_pred])[:number].index:
            good_pred += 1
        else:
            bad_pred += 1

    print(good_pred/(good_pred+bad_pred))
import pandas as pd
import numpy as np
from os import path

class Atom:
    
    def __init__(self, atom_raw):
        self._subject = atom_raw[0]
        self._predicate = atom_raw[1]
        self._objectD = atom_raw[2]
        
    def __hash__(self):
        return hash((self._subject, self._predicate, self._objectD))
    
    def __repr__(self):
        return f"{self.subject} {self.predicate} {self.objectD}"
    
    def __eq__(self, other):
        return self.subject==other.subject and self.predicate==other.predicate and self.objectD==other.objectD
        
    @property
    def subject(self):
        return self._subject
    
    @property
    def predicate(self):
        return self._predicate
    
    @property
    def objectD(self):
        return self._objectD
    
class Rule:
    
    def __init__(self, hypotheses, conclusion, otherRes):
        if not isinstance(hypotheses, tuple):
            self._hypotheses = tuple(hypotheses)
        else : 
            self._hypotheses = hypotheses
        self._conclusion = conclusion
        self._size_hypotheses = len(hypotheses)
        self._headCoverage = float(otherRes[0])
        self._stdConfidence = float(otherRes[1])
        self._pcaConfidence = float(otherRes[2])
        self._precision_train = None
        self._precision_test = None
        
        
    def __hash__(self):
        return hash((self._hypotheses, self._conclusion))
        
    def __repr__(self):
        toWrite=""
        for atom in self.hypotheses:
            toWrite += f"{atom} & "
        toWrite = toWrite[:-3] + " => " 
        toWrite += str(self.conclusion)
        return toWrite

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return (self.conclusion == other.conclusion) and (set(self.hypotheses) == set(other.hypotheses))
    
    @property
    def hypotheses(self):
        return self._hypotheses
    
    @property
    def conclusion(self):
        return self._conclusion
    
    @property
    def size_hypotheses(self):
        return self._size_hypotheses
    
    @property
    def headCoverage(self):
        return self._headCoverage
    
    @property
    def stdConfidence(self):
        return self._stdConfidence
    
    @property
    def pcaConfidence(self):
        return self._pcaConfidence
    
    @property
    def precision_train(self):
        return self._precision_train
    
    @property
    def precision_test(self):
        return self._precision_test
    
    def setPrecisionTrain(self, precision):
        self._precision_train = precision
    
    def setPrecisionTest(self, precision):
        self._precision_test = precision
        
    def toDict(self):
        return {"hypotheses":self._hypotheses, "conclusion":self._conclusion, "size_hypothese":self._size_hypotheses, "headCoverage":self._headCoverage, "stdConfidence":self._stdConfidence, "pcaConfidence":self._pcaConfidence, "precision_train": self._precision_train, "precision_test":self._precision_test}
            
# Given a feature we will compare it to a threshold      
def limit_by_threshold(X, feature, threshold):
    return X[feature] >= threshold

# Given Amie sets of rules and new parameters, it returns new set of rules describe by these parameters.
def add_parameters(amie_responses, parameters):
    new_responses = {}
    for amie_response in amie_responses:
        for para in parameters:
            
            #Optimizable
            new_response_raw = amie_responses[amie_response].copy()
            new_response = {}
            for r in new_response_raw:
                new_response[r] = new_response_raw[r].toDict()
            new_response = pd.DataFrame.from_dict(new_response, orient="index")
            
            name = amie_response
            for sub_para in para:
                if len(new_response) != 0:
                    new_response = new_response.loc[new_response.apply(func=limit_by_threshold, axis=1, feature=sub_para[0], threshold=sub_para[1])]
                if sub_para[0] == "stdConfidence":
                    name += "\n"+"stdC"+"="+str(sub_para[1])
                elif sub_para[0] == "pcaConfidence":
                    name += "\n"+"pcaC"+"="+str(sub_para[1])
                else :
                    name += "\n"+"hC"+"="+str(sub_para[1])
            new_responses[name] = new_response
    return new_responses


def save_sets_rule(root, set_rules):
    if not path.isdir(root+"/save"):
        os.mkdir(root+"/save")
    else : 
        shutil.rmtree(root+"/save")
        os.mkdir(root+"/save")
    for set_rule in set_rules:
        set_rules[set_rule].to_csv(root+"/save/"+set_rule+".tsv")
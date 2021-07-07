import pandas as pd
import numpy as np
from os import path
from subprocess import check_output
from multiprocessing import Process, Manager, SimpleQueue
import multiprocessing
import shutil
import sys
import time
from rules import *

def thread_amie(queue, name, res_rules_raw_manager, cpt):
    print(f"Process n°{name} : Launched")
    while not queue.empty():
        cv, atom, data = queue.get()
        
        res_rules_raw_manager[f"CV={cv}-A={atom}"] = check_output(f'java -jar ./../amie3.jar -const -htr approval -maxad {atom} {data}', shell=True)
        
    print(f"Process n°{name} : Finished")
        

def run_amie(atom_LIST, root, cv, const=False):
    res_rules_raw = {}
    
    if not const:
        for i in range(cv):
            for atom in atom_LIST:
                print(f"{atom}")
                data = f"{root}CV_train_{i}.tsv"
                res_rules_raw[f"CV={i}-A={atom}"] = check_output(f'java -jar ./../amie3.jar -htr approval -maxad {atom} {data}', shell=True)
    else:
        with Manager() as manager:
            q = SimpleQueue()

            for i in range(cv):
                for atom in atom_LIST:
                    q.put((i, atom, f"{root}CV_train_{i}.tsv"))
                    
            processes_to_create = multiprocessing.cpu_count()-3
            processes = list()

            res_rules_raw_manager = manager.dict()
            cpt = manager.Value("d",0)

            for name in range(processes_to_create):
                x = Process(target=thread_amie, args=(q, name, res_rules_raw_manager, cpt))
                processes.append(x)
                x.start()

            for index, process in enumerate(processes):
                process.join()

            res_rules_raw = res_rules_raw_manager.copy()
                
    return res_rules_raw
            
def parse_amie(res_rules_raw):    
    rules_per_cv = {}
    cpt = 0
    for cv_res in res_rules_raw:
        rules = set()
        for line in res_rules_raw[cv_res].decode("utf-8").split("\n"):
            if (line != "") and (line[0] == "?"):
                dic = {}
                parts = line.split("\t")

                conclusion_raw = parts[0].split("=>")[1].split("  ")
                conclusion_raw[0] = conclusion_raw[0][1:]
                dic["conclusion"] = Atom(conclusion_raw)

                hypotheses_raw = parts[0].split("=>")[0].split("  ")
                hypotheses = []
                for i in range(0, len(hypotheses_raw)-1, 3):
                    hypotheses.append(Atom(hypotheses_raw[i:i+3]))
                dic["hypotheses"] = hypotheses


                rules.add(Rule(dic["hypotheses"], dic["conclusion"], parts[1:]))
        rules_per_cv[cv_res] = rules
        cpt+=1
    return rules_per_cv
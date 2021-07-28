from rule import *

def parse_amie(res_rules_raw):    
    rules_per_cv = {}
    cpt = 0
    rules = set()
    for line in res_rules_raw.decode("utf-8").split("\n"):
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
    return rules
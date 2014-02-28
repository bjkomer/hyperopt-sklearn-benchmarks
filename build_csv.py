import csv
import sys
import re
import copy
import time

clf_dict = {"svc":{},
            "knn":{},
            "sgd":{},
            "multinomial_nb":{},
            "any":{}}

algo_dict = {"tpe":copy.deepcopy(clf_dict),
             "anneal":copy.deepcopy(clf_dict),
             "tree":copy.deepcopy(clf_dict),
             "gp_tree":copy.deepcopy(clf_dict),
             "rand":copy.deepcopy(clf_dict)}

datasets = [ "newsgroups", "convex", "mnist" ]
clfs = [ "svc", "knn", "sgd", "multinomial_nb", "any" ]
algos = [ "tpe", "anneal", "rand", "gp_tree", "tree" ]

def extract_tags( f ):
  """
  Extracts properties of the run based on the file name
  """
  #FIXME: make this less hacky and more robust
  for d in datasets:
    if d in f:
      dataset = d
      break
  for c in clfs:
    if c in f:
      clf = c
      break
  for a in algos:
    if a in f:
      algo = a
      break

  evals = f.split("_")[-2]

  return dataset, algo, clf, evals

def main( files ):
  """
  This script takes a list of result files, and creates a csv file organizing
  those results
  """
  results = {"newsgroups":copy.deepcopy(algo_dict),
             "convex":copy.deepcopy(algo_dict),
             "mnist":copy.deepcopy(algo_dict)}

  for f in files:
    dataset, algo, clf, evals = extract_tags( f )
    d = open( f, 'r' )
    d.readline()
    f1_str = d.readline()
    f1 = re.findall(r"[-+]?\d*\.\d+|\d+", f1_str)[1]
    print(f1)
    d.close()
    if evals in results[dataset][algo][clf].keys():
      results[dataset][algo][clf][evals].append(f1)
    else:
      results[dataset][algo][clf][evals] = [f1]

  with open('results.csv', 'wb') as csvfile:
    writer = csv.writer( csvfile, delimiter=',')
    for dkey, d in results.iteritems():
      for akey, a in d.iteritems():
        for ckey, c in a.iteritems():
          for ekey, e in c.iteritems():
            writer.writerow([dkey]+[akey]+[ckey]+e)

if __name__ == "__main__":
  main( sys.argv[1:] )

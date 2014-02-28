# Using hpsklearn on the 20 newsgroups dataset
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn.components import any_classifier, any_sparse_classifier, svc, \
                                 knn, sgd, tfidf, random_forest, extra_trees, \
                                 liblinear_svc, multinomial_nb

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_mldata
import mlpython.datasets.store as dataset_store

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from hyperopt import fmin, tpe, anneal, rand, hp
import hypertree.gp_tree
import hypertree.tree

import argparse

# remove headers, footers, and citations from 20 newsgroups data
REMOVE_HEADERS=False
# use the default settings up TfidfVectorizer before doing optimization
PRE_VECTORIZE=False

def score( y1, y2 ):
  length = len( y1 )
  correct = 0.0
  for i in xrange(length):
    if y1[i] == y2[i]:
      correct += 1.0
  return correct / length

# TODO: currently does not use seed for anything
def sklearn_newsgroups( classifier, algorithm, max_evals=100, seed=1,
                        filename='none.out'):

  estim = hyperopt_estimator( classifier=classifier, algo=algorithm,
                              preprocessing=[tfidf('tfidf')],
                              max_evals=max_evals, trial_timeout=300 )
  
  if REMOVE_HEADERS:
    train = fetch_20newsgroups( subset='train', 
                                remove=('headers', 'footers', 'quotes') )
    test = fetch_20newsgroups( subset='test', 
                               remove=('headers', 'footers', 'quotes') )
  else:
    train = fetch_20newsgroups( subset='train' )
    test = fetch_20newsgroups( subset='test' )
  
  
  if PRE_VECTORIZE:
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform( train.data )
    y_train = train.target

    X_test = vectorizer.transform( test.data )
    y_test = test.target

  else:

    X_train = train.data
    y_train = train.target

    X_test = test.data
    y_test = test.target

  print(y_train.shape)
  print(y_test.shape)
  find_model( X_train, y_train, X_test, y_test, estim, filename )

def sklearn_mnist( classifier, algorithm, max_evals=100, seed=1,
                   filename = 'none.out' ):

  estim = hyperopt_estimator( classifier=classifier, algo=algorithm,
                              preprocessing=[], max_evals=max_evals )

  digits = fetch_mldata('MNIST original')

  X = digits.data
  y = digits.target

  # TODO: put in option for cross-validation?
  test_size = int( 0.2 * len( y ) )
  np.random.seed( seed )
  indices = np.random.permutation(len(X))
  X_train = X[ indices[:-test_size]]
  y_train = y[ indices[:-test_size]]
  X_test = X[ indices[-test_size:]]
  y_test = y[ indices[-test_size:]]


  print(y_train.shape)
  print(y_test.shape)
  
  find_model( X_train, y_train, X_test, y_test, estim, filename )

def sklearn_convex( classifier, algorithm, max_evals=100, seed=1,
                    filename = 'none.out' ):

  estim = hyperopt_estimator( classifier=classifier, algo=algorithm,
                              max_evals=max_evals, trial_timeout=15 )

  dataset_store.download('convex')
  trainset,validset,testset = dataset_store.get_classification_problem('convex')

  X_train = trainset.data.mem_data[0]
  y_train = trainset.data.mem_data[1]
  
  X_valid = validset.data.mem_data[0]
  y_valid = validset.data.mem_data[1]
  
  X_test = testset.data.mem_data[0]
  y_test = testset.data.mem_data[1]

  print(y_train.shape)
  print(y_valid.shape)
  print(y_test.shape)
  
  find_model( X_train, y_train, X_test, y_test, estim, filename )
  
def find_model( X_train, y_train, X_test, y_test, estim, filename ):

  # The fit function does splitting into training and test sets
  estim.fit( X_train, y_train )

  pred = estim.predict( X_test )

  print( "Accuracy:" )
  print( score( pred, y_test ) ) 
  print( "F1 Score:" )
  print( metrics.f1_score( pred, y_test ) )

  print( "Model:" )
  print( estim.best_model() )
  
  f = open( filename, 'w' )
  f.write( "Accuracy: " + str( score( pred, y_test ) ) + "\n")
  f.write( "F1 Score: " + str( metrics.f1_score( pred, y_test ) ) + "\n")
  f.write( "Model: \n" )
  f.write( repr( estim.best_model() ) )
  f.close()

def main( data='newsgroups', algo='tpe', seed=1, evals=100, clf='any', text='' ):
  filename = text + algo + '_' + clf + '_' + str(seed) + '_' + str(evals) + \
             '_' + data + '.out'
  
  if algo == 'tpe':
    algorithm = tpe.suggest
  elif algo == 'anneal':
    algorithm = anneal.suggest
  elif algo == 'rand':
    algorithm = rand.suggest
  elif algo == 'tree':
    algorithm = hypertree.tree.suggest
  elif algo == 'gp_tree':
    algorithm = hypertree.gp_tree.suggest
  else:
    print( 'Unknown algorithm specified' )
    return 1
  
  # TODO: impose restrictions on classifiers that do not work on sparse data
  if clf == 'any':
    if data in ['newsgroups']:
      classifier = any_sparse_classifier('clf') 
    else:
      classifier = any_classifier('clf') 
  elif clf == 'svc':
    classifier = svc('clf') 
  elif clf == 'knn':
    classifier = knn('clf') 
  elif clf == 'sgd':
    classifier = sgd('clf') 
  elif clf == 'random_forest':
    classifier = random_forest('clf') 
  elif clf == 'extra_trees':
    classifier = extra_trees('clf') 
  elif clf == 'liblinear_svc':
    classifier = liblinear_svc('clf') 
  elif clf == 'multinomial_nb':
    classifier = multinomial_nb('clf') 
  else:
    print( 'Unknown classifier specified' )
    return 1
  
  if data == 'newsgroups':
    sklearn_newsgroups( classifier=classifier, algorithm=algorithm, 
                        max_evals=evals, seed=seed, filename=filename)
  elif data == 'convex':
    sklearn_convex( classifier=classifier, algorithm=algorithm, 
                        max_evals=evals, seed=seed, filename=filename)
  elif data == 'mnist':
    sklearn_mnist( classifier=classifier, algorithm=algorithm, 
                        max_evals=evals, seed=seed, filename=filename)
  else:
    print( "Unknown dataset specified" )

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser( description='run a classifier on a dataset using hyperopt sklearn' )
  
  #TODO: add preprocessing as an option
  #TODO: add timeout as an option
  parser.add_argument('--data', '-d', dest='data', default='newsgroups',
                      help='dataset to use, one of: [newsgroups, convex, mnist]')
  parser.add_argument('--algo', '-a', dest='algo', default='rand',
                      help='optimization algorithm to use, one of: [rand, anneal, tpe, tree, gp_tree]')
  parser.add_argument('--clf', '-c', dest='clf', default='any',
                      help='classifier to use')
  parser.add_argument('--seed', '-s', dest='seed', default=1,
                      help='random seed to use')
  parser.add_argument('--evals', '-e', dest='evals', default=100,
                      help='the number of evaluations in the search space')
  parser.add_argument('--text', '-t', dest='text', default='',
                      help='optional text to append to the output file name')
  
  args = parser.parse_args()

  main( data=args.data, algo=args.algo, clf=args.clf, seed=int(args.seed),
        evals=int(args.evals), text=args.text )

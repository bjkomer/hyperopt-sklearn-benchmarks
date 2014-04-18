#TODO: put in preprocessing
#Runs baseline tests using the default values for different classifiers in sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_mldata
try:
  import mlpython.datasets.store as dataset_store
  CONVEX_EXISTS=True
except:
  CONVEX_EXISTS=False

REMOVE_HEADERS=False

def newsgroups():

  if REMOVE_HEADERS:
    print( "Headers Removed\n" )
    train = fetch_20newsgroups( subset='train', 
                                remove=('headers', 'footers', 'quotes') )
    test = fetch_20newsgroups( subset='test', 
                               remove=('headers', 'footers', 'quotes') )
  else:
    print( "No Data Removed\n" )
    train = fetch_20newsgroups( subset='train' )
    test = fetch_20newsgroups( subset='test' )

  vectorizer = TfidfVectorizer()
  X_train = vectorizer.fit_transform( train.data )
  y_train = train.target

  X_test = vectorizer.transform( test.data )
  y_test = test.target

  """
  # Take into account the fact that hpsklearn only trains on 80% of the train data
  data_length = X_train.shape[0]
  p = np.random.RandomState(123).permutation( data_length )
  n_fit = int(.8 * data_length)
  Xfit = X_train[p[:n_fit]]
  yfit = y_train[p[:n_fit]]
  Xval = X_train[p[n_fit:]]
  yval = y_train[p[n_fit:]]
  """
  clfs = [ MultinomialNB(), SVC(),
           KNeighborsClassifier(),
           SGDClassifier() ]
  
  print(" 20 newsgroups\n")
  with open( "newsgroup_baselines.txt", 'w' ) as f:
    for clf in clfs:
      clf.fit( X_train, y_train )
      #clf.fit( Xfit, yfit )
      pred = clf.predict( X_test )
      score = metrics.f1_score( y_test, pred )
      print( "Classifier: %s\nScore: %f\n" % (clf, score) )
      f.write("Classifier: %s\nScore: %f\n\n" % (clf, score))

def mnist():

  digits = fetch_mldata('MNIST original')

  X = digits.data
  y = digits.target

  #test_size = int( 0.2 * len( y ) )
  test_size = 10000
  np.random.seed( 13 )
  indices = np.random.permutation(len(X))
  X_train = X[ indices[:-test_size]]
  y_train = y[ indices[:-test_size]]
  X_test = X[ indices[-test_size:]]
  y_test = y[ indices[-test_size:]]
  
  pca = PCA()
  X_train_pca = pca.fit_transform( X_train )
  X_test_pca = pca.fit_transform( X_test )
  
  clfs = [ MultinomialNB(), SVC(),
           KNeighborsClassifier(),
           SGDClassifier() ]
  
  pca_clfs = [ SVC(),
               KNeighborsClassifier(),
               SGDClassifier() ]
  
  print("MNIST\n")
  with open( "mnist_baselines.txt", 'w' ) as f:
    for clf in clfs:
      clf.fit( X_train, y_train )
      pred = clf.predict( X_test )
      score = metrics.f1_score( y_test, pred )
      print( "Classifier: %s\nScore: %f\n" % (clf, score) )
      f.write("Classifier: %s\nScore: %f\n\n" % (clf, score))
    for clf in pca_clfs:
      clf.fit( X_train_pca, y_train )
      pred = clf.predict( X_test_pca )
      score = metrics.f1_score( y_test, pred )
      print( "Classifier: PCA + %s\nScore: %f\n" % (clf, score) )
      f.write("Classifier: PCA + %s\nScore: %f\n\n" % (clf, score))

def convex():

  dataset_store.download('convex')
  trainset,validset,testset = dataset_store.get_classification_problem('convex')

  X_train = trainset.data.mem_data[0]
  y_train = trainset.data.mem_data[1]
  
  X_valid = validset.data.mem_data[0]
  y_valid = validset.data.mem_data[1]
  
  X_test = testset.data.mem_data[0]
  y_test = testset.data.mem_data[1]
  
  X_fulltrain = np.concatenate((X_train, X_valid))
  y_fulltrain = np.concatenate((y_train, y_valid))
  
  pca = PCA()
  X_train_pca = pca.fit_transform( X_fulltrain )
  X_test_pca = pca.fit_transform( X_test )
  
  clfs = [ MultinomialNB(), SVC(),
           KNeighborsClassifier(),
           SGDClassifier() ]
  
  pca_clfs = [ SVC(),
               KNeighborsClassifier(),
               SGDClassifier() ]
  
  print("Convex\n")
  with open( "convex_baselines.txt", 'w' ) as f:
    for clf in clfs:
      clf.fit( X_fulltrain, y_fulltrain )
      pred = clf.predict( X_test )
      score = metrics.f1_score( y_test, pred )
      print( "Classifier: %s\nScore: %f\n" % (clf, score) )
      f.write("Classifier: %s\nScore: %f\n\n" % (clf, score))
    for clf in pca_clfs:
      clf.fit( X_train_pca, y_fulltrain )
      pred = clf.predict( X_test_pca )
      score = metrics.f1_score( y_test, pred )
      print( "Classifier: PCA + %s\nScore: %f\n" % (clf, score) )
      f.write("Classifier: PCA + %s\nScore: %f\n\n" % (clf, score))

if __name__ == "__main__":
  #newsgroups()
  mnist()
  if CONVEX_EXISTS:
    convex()
  else:
    print("Convex dataset could not be found")

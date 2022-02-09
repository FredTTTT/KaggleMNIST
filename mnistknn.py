import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import CSV et mise en forme pour classifieur
data = pd.read_csv('train.csv')
target = pd.read_csv('train.csv',usecols=[0])
target = np.ravel(target)
data.drop('label', inplace=True, axis=1)

# réduction du training set
sample = np.random.randint(42000, size=5000)
data = data.values[sample]
target = target[sample]

# séparation training test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)

# classifieur PPV
# brut de fonderie, PPV (plus proches voisins) efficace mais très lourd en mémoire ce qui explique la réduction du training set ci-dessus
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)

# performance sur test
accuracy = knn.score(xtest, ytest)
print('Accuracy 4-NN: %f' % accuracy)

# on applique sur les données test de kaggle 
# nota : le test.csv de kaggle est mal nommé, il aurait fallu qu'il s'appelle eval.csv popur bien distinguer les parties train, test et eval
# en effet, la partie normalement dénommé test permet au développeur d'évaluer son algo puisque qu'il connait la VT (vérité terrain) contrairement au lot eval
eval = pd.read_csv('test.csv')
values = eval.values
predicted = knn.predict(values)

# on écrit le csv pour submission
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['Label'] = predicted
sample_submission.to_csv("submission.csv",index=False)
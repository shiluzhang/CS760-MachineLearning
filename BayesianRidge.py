
import os
import pandas as pd
import sys
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import BayesianRidge

## distance-stratified correlation:
def distance_stratified_correlation(distance,ytest,ypred):
    corr=np.zeros(len(distance))
    for dist in distance:
        itemindex = np.where(distance==dist)
        k=int(dist/5000)
        corr[k]=stats.pearsonr(ytest[itemindex],ypred[itemindex])[0]
    return corr

# Plot Distance stratified correlation:
def plot_distance_stratified_correlation(image_name,distance,ytest,ypred):
    corr=distance_stratified_correlation(distance,ytest,ypred)
    fig = plt.figure()
    plt.plot(distance, corr, '-')
    plt.xlabel('Distance')
    plt.ylabel('Correlation')
    plt.title('Distance stratified correlation curve')
    fig.savefig(image_name)

# Plot scatterplot
def plot_scatterplot(image_name,ytest,ypred):
    fig = plt.figure()
    plt.scatter(ypred, ytest,s=3)
    ymax=np.ceil(np.max([ypred,ypred]))
    plt.xlim(0, ymax)
    plt.ylim(0, ymax)
    plt.xlabel('Predicted count')
    plt.ylabel('True count')  
    plt.title('Scatterplot of Predicted count vs True count')
    fig.savefig(image_name)

## Bayesian ridge regression:
os.chdir('BayesianRidge/')
yprediction=[]
distance=[]
ytest=[]
## 5-folds Cross Validation:
for i in range(0,5):
    trainfile='./train'+str(i)+'_upto200k.txt'
    testfile='./test'+str(i)+'_upto200k.txt'

    training = pd.read_table(trainfile, sep='\t')
    test = pd.read_table(testfile, sep='\t')

    data=pd.DataFrame.as_matrix(training)
    testdata=pd.DataFrame.as_matrix(test)

    y=data[:,30]
    x=data[:,1:30]
    ytest=np.concatenate((testdata[:,30],ytest), axis=0)
    xtest=testdata[:,1:30]
    distance=np.concatenate((testdata[:,29],distance), axis=0)
    clf = BayesianRidge(compute_score=True,normalize=True)
    clf.fit(x, y)
    clf.coef_
    ypred=clf.predict(xtest)
    yprediction = np.concatenate((ypred, yprediction), axis=0)
    ypredall.append(ypred)

## output predictions:
pred=pd.DataFrame({'Distance':distance,'TrueValue':ytest,'Predicted':yprediction})
pred.to_csv('Bayesian_ridge_regression_Predictions.txt', index = False,header=True,  sep='\t', mode='w')


## Evaluation:
correlation=stats.pearsonr(ytest,ypred)[0]
print('Correlation is: '+str(correlation))

image_name = "Bayesian_ridge_regression_scatterplot.pdf"
plot_scatterplot(image_name,ytest,ypred)

image_name = "Bayesian_ridge_regression_distance_stratified_correlation.pdf"
plot_distance_stratified_correlation(image_name,distance,ytest,yprediction)








#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from matplotlib import cm
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_wine
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.model_selection import cross_val_score


PRINT = False

def Apply_and_plot_Knn(Ks,X_train,y_train,X_test,y_test,nrow,ncols,test):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    fig, sub = plt.subplots(nrows = nrow, ncols = ncols,figsize=(4,4))
    fig.subplots_adjust(hspace=0.7, wspace=0.7)
    accuracies =[]
    
    if np.ndim(sub) > 1:
        sub = sub.flatten()
    else:
        sub = [sub]
    
    for i,(k,ax) in enumerate(zip(Ks, sub)): 
        
        knn = KNeighborsClassifier(n_neighbors = k)
        #Train the model using the training sets
        knn.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = knn.predict(X_test)

        accuracies.append(metrics.accuracy_score(y_test, y_pred))


        h = .02 # step size in the mesh
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        if(test == 0):
            ax.scatter(X_train[:,0], X_train[:,1],c=y_train,  cmap=cmap_bold,
                        edgecolor='k', s=20 )
        else:
            ax.scatter(X_test[:,0], X_test[:,1],c=y_test,  cmap=cmap_bold,
                    edgecolor='k', s=20 )
   
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        ax.set_title("k = %.0f" %k + " Accuracy of: %.2f " % accuracies[i] )
    return accuracies

def SVM(X,y,X_test_a,y_test_a,kernel,C, Gamma_best):
   

    linearSVM = svm.SVC(C = C,gamma=Gamma_best,kernel=kernel)
    linearSVM.fit(X, y) # fit on training data
    Y_pred = linearSVM.predict(X_test_a) # predict evaluation data
    accuracy = linearSVM.score(X_test_a, y_test_a) # get accuracy level
    print("C =", C) 
    print("---> Accuracy = %.2f" % accuracy)
    return linearSVM, accuracy


def plot_model(X,y,X_test,y_test,clf,ctype,title,ax):
    # create a mesh to plot in
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #plt.figure()
    #plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
   
    # Plot also the training points
    ax.scatter(X_test[:,0], X_test[:,1],c=y_test,  cmap=plt.cm.coolwarm,
                edgecolor='k', s=20 )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    ax.set_title(title)
   
    
def plot_accuracy(value,accuracies,axes_name,scale):
    
    
    plt.subplots(figsize=(3,3))
    plt.ylabel('accuracy')
    plt.xlabel(axes_name)
    plt.xticks(np.array(value))
    
    plt.plot(value,accuracies,'bo-')
    if scale == 1:
        plt.xscale('log')
    plt.grid(axis='y')
    plt.show()

def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv = nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_



def applySVM_C_Gamma(kernel, C_list, Gamma_list, X_train, Y_train, X_test, Y_test):
   
    models = [] # list of all the different SVM that we had trained
    
    highest_accuracy = -1 # the highest accuracy found
    
    hyperparameters = np.empty(shape=(len(C_list), len(Gamma_list))) # matrix of accuracy for each combination of hyperparameters

    for c in C_list:
        for gamma in Gamma_list:

            if PRINT == True:
                print("C =", c, "and Gamma =", gamma)
            
            linearSVM = svm.SVC(C = c, gamma=gamma, kernel=kernel)
            linearSVM.fit(X_train, Y_train.ravel()) # fit on training data
            Y_predicted = linearSVM.predict(X_test) # predict evaluation data
            accuracy = linearSVM.score(X_test, Y_test) # get accuracy level
            models.append(linearSVM) # save model so that we can plot later

            
            if PRINT == True:
                print("Accuracy : %.3f" % accuracy, "\n")
    
            if(accuracy > highest_accuracy): 
                highest_accuracy = accuracy
    
            hyperparameters[C_list.index(c), Gamma_list.index(gamma)] = accuracy # save accuracy 
        

    return models, hyperparameters, highest_accuracy


def SVM_C_Gamma_Kfold(kernel, C_list, Gamma_list, X_train, Y_train, kfold):
    
    models = [] # list of all the different SVM that we had trained
    
    highest_accuracy = -1 # the highest accuracy found
    
    hyperparameters = np.empty(shape=(len(C_list), len(Gamma_list))) 

    for c in C_list:
        for gamma in Gamma_list:

            if PRINT == True:
                print("C =", c, "and Gamma =", gamma)
            
            clf = svm.SVC(C = c, gamma=gamma, kernel=kernel)
            scores = cross_val_score(clf,X_train, Y_train, cv=kfold)
            
            
            accuracy = scores.mean()
            
            if PRINT == True:
                print("Mean accuracy: %.3f" % accuracy, "\n")
    
            if(accuracy > highest_accuracy): 
                highest_accuracy = accuracy
    
            hyperparameters[C_list.index(c), Gamma_list.index(gamma)] = accuracy # save accuracy 

    return models, hyperparameters, highest_accuracy


def plotHeat(hyperparameters,C_list, Gamma_list ):

    fig, axes = plt.subplots(figsize=(4,4))
    sns.heatmap(hyperparameters, cmap='Purples', xticklabels=C_list,yticklabels=Gamma_list, annot=True, 
                        ax  = axes, square = True) 

    axes.invert_yaxis()
    axes.set_xlabel('Gamma')
    axes.set_ylabel('C')
    axes.set_title('Accuracy C and gamma')


def Find_BestHyperparameters(hyperparameters,C_list,Gamma_list,accuracy) :
    
    C_bests_list = []
    Gamma_bests_list = []

    for c in C_list:
        for gamma in Gamma_list:
            if(hyperparameters[C_list.index(c), Gamma_list.index(gamma)] == accuracy): 
                print("C=", c, ", Gamma=", gamma, "accuracy ", accuracy)
                C_bests_list.append(c) # get C values that give maximum accuracy
                Gamma_bests_list.append(gamma) #  corresponding gamma values


    C_best = min(C_bests_list)

    Gamma_best = Gamma_bests_list[C_bests_list.index(C_best)]
    
    return  C_best,Gamma_best


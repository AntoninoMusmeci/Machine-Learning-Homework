#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# We want to analyze the Wine dataset that contains the results of a chemical analysis of wines grown in a specific area of Italy. The dataset contains 178 examples with 13 features  
# 
# Alcohol  
# Malic acid  
# Ash  
# Alcalinity of ash  
# Magnesium  
# Total phenols  
# Flavanoids  
# Nonflavanoid phenols  
# Proanthocyanins  
# Color intensity  
# Hue  
# OD280/OD315 of diluted wines  
# Proline  
# 
# 
# Number of instances of each wine class  
# 
# Class 1 - 59  
# Class 2 - 71  
# Class 3 - 48  
# 
# The goal of the analysis is build a classification  model using first K-Nearest Neighbors and then support vector machine to determine from which of the three cultivators the wine has come from

# In[1]:


# Import functions and library
from myfunc import *
wine_dataset = load_wine() #load the dataset
data = pd.DataFrame(data=wine_dataset.data,columns=wine_dataset.feature_names)
data.head()


# 
# Firstly we need to split into training 50%, validation 20% and test 30% sets  and standardize it.
# 
# Before starting the analysis, we must standardize the data. It deals with transforming the data that was acquired in different formats to a standard format. This  in general is needed to obtain  best results for  ML algorithms. Standardize means to scale the features such that they have zero as mean and one as variance. 
# The standard score of a sample x is calculated as:
# 
# $$x_{std} = (x-u)/s$$
# 
# where "$x$" is a feature value, "$u$" is the mean and "$s$" is the variance of the input features.
# 
# We can use sklearn.preprocessing.StandardScaler to perform standardization
# 
# Then we consider only the first two features for a 2D representation of the image and discard the others

# In[2]:


# wine_dataset = load_wine() #load the dataset

X = wine_dataset.data  #features
y = wine_dataset.target #labels

#we divide the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=42)

scaler = preprocessing.StandardScaler() 
scaler.fit(X_train) 


X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

# we divide the training set into training and validation set
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=2/7, random_state=42)


X_train_2D = X_train[:,:2]
X_test_2D = X_test[:,:2]
X_val_2D = X_val[:,:2]


# 
# <h2>K-Nearest Neighbors</h2>  
# 
# The K-Nearest Neighbors algorithm (KNN) is a non-parametric method, which considers the K closest training examples to the point of interest for predicting its class. This is done by a simple majority vote over the K closest points.
# 
# In the classification phase, k is a user-defined constant. The best choice of k depends upon the data; generally, larger values of k reduces effect of the noise on the classification, but make boundaries between classes less distinct.To choose a good k value we use the validation set to score the models fitted one the training set and we select the k value that give the best accuracy

# In[3]:


Ks=[1,3,5,7]
accuracies_knn = Apply_and_plot_Knn(Ks,X_train_2D,y_train,X_val_2D,y_val,2,2,0)


# In[4]:


plot_accuracy(Ks,accuracies_knn,'k value',0)
#select best k value
best_k = Ks[np.argmax(accuracies_knn)]
print("The value of k that gives the higher accuracy is k =", best_k)


# In[5]:


A = Apply_and_plot_Knn([best_k],X_train_2D,y_train,X_test_2D,y_test,1,1,1)
print("The accuracy of the model on the test set is %.2f" % A[0])


# 
# <h2>Support vector machine (SVM) </h2> 
# 
# <h3>Linear SVM</h3> 
# 
# A Support Vector Machine (SVM) is a discriminative classifier defined by a separating hyperplane. The aim of 
# Support Vector classification is to find 'good' separating hyperplanes in a high dimensional feature space, where a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class
# 
# 
# In many real-world problems data are noisy and there will in general be no linear separation in the feature space
# 
# When the two classes are not linearly separable, the condition for the optimal hyper-plane can be relaxed by including an extra term: 
# 
# $$y_i ({\bf x}_i^T{\bf w} +b) \ge 1-\xi_i,\;\;\;(i=1,\cdots,m)$$
# 
# For minimum error, $\xi_i \ge 0$ should be minimized as well as $\vert\vert{\bf w}\vert\vert$, and the objective function becomes:
# 
# 
#   \begin{eqnarray}
# &	\mbox{minimize} & {\bf w}^T {\bf w}+C\sum_{i=1}^m \xi_i^k
# 	\nonumber \\
# &	\mbox{subject to} & y_i ({\bf x}_i^T {\bf w}+b) \ge 1-\xi_i,
# 	\;\;\;\mbox{and}\;\;\;\xi_i \ge 0;\;\;\;(i=1,\cdots,m)
# 	\nonumber
# \end{eqnarray}
# 
# Here $C$ is a regularization parameter that controls the trade-off between maximizing the margin and minimizing the training error. 
# The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points. 
# 
# 

# In[6]:


C = [0.001, 0.01, 0.1, 1, 10, 100,1000]


# Now we'll train and evaluate a model for each value of C in the list. We want to choose the C that gives the highest accuracy on the validation set.
# 
# 

# In[7]:


models =[]
accuracies = []

for c in C:
    model, accuracy = SVM(X_train_2D,y_train,X_val_2D,y_val,'linear',c,'auto')
    models.append(model)
    accuracies.append(accuracy)


# In[8]:


fig, sub = plt.subplots(nrows = 3, ncols = 3,figsize=(7,7))
fig.subplots_adjust(hspace=0.6, wspace=0.6)
for i,(clf,ax) in enumerate(zip(models, sub.flatten())): 
    plot_model(X_train_2D,y_train,X_train_2D,y_train,clf,
               'coolwarm',"C = " + str(C[i]), ax)
    


# The plots above show the effect the parameter C. If we choose a small margin (large C), we don't trust that our data are well separated, so it will be difficult to classify them, and in this case, a small margin will help. But if the margin is too small, it could not be possible to separate the classes using too few samples as support vectors.

# In[9]:


plot_accuracy(C,accuracies,'C value',1)


# 
# As we can see from the above chart, the accuracy is increasing for higher values of C, the model with the highest accuracy is in fact a small margin SVM. If there are more values of C that give higher accuracy, we choose the smaller one which is the one that gives the larger margin

# In[10]:


ind = np.argmax(accuracies)
best_C = C[ind]
best_clf,a = SVM(X_train_2D,y_train,X_test_2D,y_test,'linear',best_C,'auto')
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_2D,y_train,X_test_2D,y_test,best_clf,'coolwarm',"C = " + str(C[ind]),sub)


# <h3> RBF Kernel </h3>
# 
# The function of kernel is to take data as input and transform it into the required form. 
# The kernel functions return the inner product between two points in a suitable feature space. Thus by defining a notion of similarity, with little computational cost even in very high-dimensional spaces.
#  
# Now we will use the RBF kernel, that is one of the most used,instead of the linear one  
#  $$  K(x,x′)=exp(\lambda||x−x′||^2) $$
# 
# As for the linear, we will train and evaluate a model for each value of C and we will choose the C that gives the highest accuracy.
# 

# In[11]:


models_rbf = []
accuracies_rbf = []

for c in C:
    model_rbf, accuracy_rbf = SVM(X_train_2D,y_train,X_val_2D,y_val,'rbf',c,'auto')
    models_rbf.append(model_rbf)
    accuracies_rbf.append(accuracy_rbf)


# In[12]:


fig, sub = plt.subplots(nrows = 3, ncols = 3,figsize=(7,7))
fig.subplots_adjust(hspace=0.6, wspace=0.6)

for i,(clf,ax) in enumerate(zip(models_rbf, sub.flatten())): 
    plot_model(X_train_2D,y_train,X_train_2D,y_train,clf,'coolwarm',"C = " + str(C[i]),ax)


# In[13]:


plot_accuracy(C,accuracies_rbf,'C value',1)


# As for the linear, if there are more values of C that give higher accuracy, we choose the smaller one 

# In[14]:


ind = np.argmax(accuracies_rbf)
best_C_rbf = C[ind]
best_clf,a =SVM(X_train_2D,y_train,X_test_2D,y_test,'rbf',best_C_rbf,'auto')
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_2D,y_train,X_test_2D,y_test,best_clf,'coolwarm',"C = " + str(best_C_rbf),sub)


# <h3> Grid search C and Gamma </h3>
# 
# This time we have to find the best pair C and gamma.
# The gamma parameter defines how far the influence of a single training example reaches. This means that high Gamma will consider only points close to the plausible hyperplane and low Gamma will consider points at greater distance. the gamma parameter can be said to adjust the curvature of the decision boundary.
# 

# In[15]:


C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 

Gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000] 

models, hyperparameters, highest_accuracy = applySVM_C_Gamma('rbf', C_list, Gamma_list,
             X_train_2D, y_train, X_val_2D, y_val,) 


# In[16]:


plotHeat(hyperparameters,C_list, Gamma_list)

print("The pairs that give highest accuracy are: ")
C_best,Gamma_best = Find_BestHyperparameters(hyperparameters,C_list,Gamma_list,highest_accuracy)  


# The heatmap above shows the accuracy on the validation set for each pair C and Gamma.
# There is a region of the heatmap populated by high accuracy values. We can see that high value of Gamma doesn't working very well to classify our samples, even if we consider an high value for C.

# In[17]:


print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)

best_clf,accuracy =SVM(X_train_2D,y_train,X_test_2D,y_test,'rbf',C_best,Gamma_best)
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_2D,y_train,X_test_2D,y_test,best_clf,
           'coolwarm',"C = %.2f " % C_best + " Gamma = %.2f " % Gamma_best + "\n Accuracy = %.2f" % accuracy,sub)


# We obtained a lower  accuracy than that we had obtained without grid search on gamma. We are evaluating the models on only 20% of the available samples, with a small dataset like the one we used, this means that we are using too less data to decide which values of the hyperparameters to use to tune our models.
# 

# <h3>  K-Fold </h3>
#     
# Now we'll use the K-fold crossvalidation technique to validate our models. As before, we want to find the values of C and Gamma that give us the highest accuracy.
# We merge the training and validation split. We  now have 70% training and 30% test data. The training set will then be splitted into a number of "K" different sets, and we will perform K rounds of training and evaluation, each time training on K-1 folds, and testing on the remaining fold.
# 
# This will allow to perform the validation more in depth, by using more samples. We have a chance that by doing this we'll tune the model with the right parameters.

# In[18]:


from sklearn.model_selection import cross_val_score

X_train_Kfold = np.concatenate((X_train_2D, X_val_2D))
y_train_Kfold = np.concatenate((y_train,y_val))


models, hyperparameters, highest_accuracy = SVM_C_Gamma_Kfold('rbf', C_list, Gamma_list, 
                     X_train_Kfold, y_train_Kfold, 5) 


# In[19]:


plotHeat(hyperparameters,C_list, Gamma_list)
C_best,Gamma_best = Find_BestHyperparameters(hyperparameters,C_list,Gamma_list,highest_accuracy) 


# In[20]:


print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)

best_clf,accuracy =SVM(X_train_Kfold,y_train_Kfold,X_test_2D,y_test,'rbf',C_best,Gamma_best)
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_Kfold,y_train_Kfold,X_test_2D,y_test,best_clf,
           'coolwarm',"C = %.3f " % C_best + " Gamma = %.3f " % Gamma_best + "\n Accuracy = %.2f" % accuracy,sub)


# In K-NN the sample is given a classification based only on the nearby instances, and anything farther away is ignored.  
# K-NN can generate a highly convoluted decision boundary as it is driven by the raw training data.
# SVM on the other hand, attempts to find a hyper-plane separating the different classes, with the maximum margin.
# This means that create a large margin between data points and the decision boundary. 
# The most important training instances are the ones on the boundaries. 
# The tradeoff between how many instances to allow on the wrong side and how complex the boundary is controls how complex the model is. 
# The SVM is generally more complex becouse take more information into account when classifying the target instance and will generally have much better accuracy.

# <h2> Analysis with other features pair </h2>

# In[21]:


data = pd.DataFrame(data=X_train,columns=wine_dataset.feature_names)

data['target']=y_train
# data['class']=data['target'].map(lambda ind: wine_dataset.target_names[ind])
means = data.groupby('target').mean() #mean for each features and class
std = data.groupby('target').std() #std for each features and class


# the features that most helps discriminate among the three class is the one that has the most distant means and the smallest standard deviations. This corresponds to distributions of values that are well separated and all fall within a short distance on the mean value.

# In[22]:


from scipy.stats import norm
colors = ['b','g','r']
k = 0 
fig, sub = plt.subplots(nrows = 4, ncols = 4,figsize=(10,10))
fig.subplots_adjust(hspace=1, wspace=1) 
sub = sub.flatten()
for i, n in enumerate(wine_dataset.feature_names):
    
    for c, color in enumerate(colors):
        filter = data['target'] == c
#         values = data[n].where(filter) 
       
#         plt.hist(values, density=True, alpha=0.2, color=color)
        u = means[n][c]
        s = std[n][c]
        x = np.linspace(u-5*s, u+5*s, 100)
        sub[k].plot(x, norm(u,s).pdf(x), color=color)
        sub[k].set_xlabel(f"{n}")
        sub[k].set_ylabel("density")
    k = k+1   


# We use the last 2 features to try to achieve greater accuracy

# In[23]:


X_train_2D = X_train[:,11:13]
X_test_2D = X_test[:,11:13]
X_val_2D = X_val[:,11:13]


# <h2> KNN </h2>

# In[24]:


Ks=[1,3,5,7]
accuracies_knn = Apply_and_plot_Knn(Ks,X_train_2D,y_train,X_val_2D,y_val,2,2,0)


# In[25]:


plot_accuracy(Ks,accuracies_knn,'k value',0)
#select best k value
best_k = Ks[np.argmax(accuracies_knn)]
print("The value of k that gives the higher accuracy is k =", best_k)


# In[26]:


A = Apply_and_plot_Knn([best_k],X_train_2D,y_train,X_test_2D,y_test,1,1,1)
print("The accuracy of the model on the test set is %.2f" % A[0])


# <h2> Linear SVM </h2>

# In[27]:


C = [0.001, 0.01, 0.1, 1, 10, 100,1000]
models =[]
accuracies = []

for c in C:
    model, accuracy = SVM(X_train_2D,y_train,X_val_2D,y_val,'linear',c,'auto')
    models.append(model)
    accuracies.append(accuracy)


# In[28]:


ind = np.argmax(accuracies)
best_C = C[ind]
best_clf,a = SVM(X_train_2D,y_train,X_test_2D,y_test,'linear',best_C,'auto')
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_2D,y_train,X_test_2D,y_test,best_clf,'coolwarm',"C = " + str(C[ind]),sub)


# <h3> Grid search and K-fold </h3>

# In[29]:


C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 

Gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000] 
X_train_Kfold = np.concatenate((X_train_2D, X_val_2D))
y_train_Kfold = np.concatenate((y_train,y_val))


models, hyperparameters, highest_accuracy = SVM_C_Gamma_Kfold('rbf', C_list, Gamma_list, 
                     X_train_Kfold, y_train_Kfold, 5)
plotHeat(hyperparameters,C_list, Gamma_list)
C_best,Gamma_best = Find_BestHyperparameters(hyperparameters,C_list,Gamma_list,highest_accuracy) 


# In[30]:


print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)

best_clf,accuracy =SVM(X_train_Kfold,y_train_Kfold,X_test_2D,y_test,'rbf',C_best,Gamma_best)
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_Kfold,y_train_Kfold,X_test_2D,y_test,best_clf,
           'coolwarm',"C = %.3f " % C_best + " Gamma = %.3f " % Gamma_best + "\n Accuracy = %.2f" % accuracy,sub)


# <h2> PCA </h2>
# 
# We use PCA to account for as much of the variability in the data as possible

# In[31]:


pca = PCA(2) # get first two principal components
pca.fit(X_train)
X_train_PCA = pca.transform(X_train)
X_val_PCA =pca.transform(X_val)
X_test_PCA = pca.transform(X_test)


# In[32]:


fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.scatter(X_train_PCA[:,0],X_train_PCA[:,1], c=y_train)
ax.set_title('first 2 principal component')
ax.grid()

plt.tight_layout()
plt.show()


# The chart above show that the first 2 principal components the data can be clearly separated into 3 classes so we should get a higher accuracy

# <h2> KNN </h2>

# In[33]:


Ks=[1,3,5,7]
accuracies_knn = Apply_and_plot_Knn(Ks,X_train_PCA,y_train,X_val_PCA,y_val,2,2,0)


# In[34]:


plot_accuracy(Ks,accuracies_knn,'k value',0)
#select best k value
best_k = Ks[np.argmax(accuracies_knn)]
print("The value of k that gives the higher accuracy is k =", best_k)


# In[35]:


A = Apply_and_plot_Knn([best_k],X_train_PCA,y_train,X_test_PCA,y_test,1,1,1)
print("The accuracy of the model on the test set is %.2f" % A[0])


# In[36]:


C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 

Gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000] 
X_train_Kfold = np.concatenate((X_train_PCA, X_val_PCA))
y_train_Kfold = np.concatenate((y_train,y_val))


models, hyperparameters, highest_accuracy = SVM_C_Gamma_Kfold('rbf', C_list, Gamma_list, 
                     X_train_Kfold, y_train_Kfold, 5)
plotHeat(hyperparameters,C_list, Gamma_list)
C_best,Gamma_best = Find_BestHyperparameters(hyperparameters,C_list,Gamma_list,highest_accuracy) 


# In[37]:


print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)

best_clf,accuracy =SVM(X_train_Kfold,y_train_Kfold,X_test_PCA,y_test,'rbf',C_best,Gamma_best)
fig, sub = plt.subplots(nrows = 1, ncols = 1,figsize=(4,4))
plot_model(X_train_Kfold,y_train_Kfold,X_test_PCA,y_test,best_clf,
           'coolwarm',"C = %.3f " % C_best + " Gamma = %.3f " % Gamma_best + "\n Accuracy = %.2f" % accuracy,sub)


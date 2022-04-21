from time import sleep
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import streamlit as st
import datetime 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from bia_functions import add_params_classifier, cleaning_dataset, get_classifier, normalize, put_dataset, solve, user_input_features, get_grid_rf
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV

TYPE_OF_PROBLEM = ['Classification', 'Regression']
CLASSIFIERS = ['KNN', 'SVM', 'Random Forest', 'Markov Models']

with st.spinner("Loading dataset..."):
    st.sidebar.header('User Input Parameters')
    X, y = put_dataset()
    X['explicit'] = X['explicit'].map({True:1, False:0}, na_action=None)
with st.spinner("Loading sidebar..."):
    classifier_name, type_of_problem, features_to_remove = user_input_features(X, TYPE_OF_PROBLEM, CLASSIFIERS)
with st.spinner("Cleaning and normalizing dataset..."):
    X = cleaning_dataset(X, features_to_remove)
    X_norm = normalize(X)
with st.spinner("Setting up classifier..."):
    params = add_params_classifier(classifier_name)
    classifier = get_classifier(classifier_name, params, type_of_problem)
with st.spinner("Training the model..."):
    X_train, X_test, y_train, y_test, y_pred = solve(X, y, classifier, classifier_name)
    

st.sidebar.download_button(label = 'Download dataset', data = X.to_csv(index=False), file_name='songs.csv')

with st.spinner("Feature engineering..."):
    st.header('FEATURE ENGINEERING')
    st.subheader('The dataset: ')
    st.dataframe(X)
    st.subheader('Statistics about the dataset: ')
    st.dataframe(X.describe())
    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
with st.spinner("Correlation pearson..."):
    st.subheader('Correlation Matrix: ')
    corr_matrix = X.corr(method='pearson')
    corr_matrix.style.background_gradient(cmap='coolwarm')
    st.write(corr_matrix)
    
with st.spinner("Correlation MAP..."):
    st.subheader('Correlation MAP: ')
    #correlation map
    f,ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
    st.write(f)
    
with st.spinner("Confusion matrix..."):
    st.subheader('Confusion matrix: ')
    #f,ax = plt.subplots(figsize=(10, 5))
    cm = confusion_matrix(y_test,classifier.predict(X_test))
    #sns.heatmap(cm,annot=True,fmt="d", ax=ax)
    st.write(cm)


with st.spinner("Finding bests features..."):
    # find best scored 5 features
    st.subheader('Finding bests features:')
    select_feature = SelectKBest(f_classif, k=5).fit(X_train, y_train)
    scores = pd.concat([pd.DataFrame(data=X_train.columns),pd.DataFrame(data=select_feature.scores_[:])],axis=1)
    scores.columns = ['cat','score']
    scores = scores.sort_values('score',ascending=False)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='score',y='cat',data=scores, palette='seismic', ax=ax)
    plt.show()
    st.write(fig)
    

with st.spinner("Finding optimal number of features and best ones..."):
    st.subheader('Finding optimal number of features:')
    # The "accuracy" scoring is proportional to the number of correct classifications
    rfecv = RFECV(estimator=classifier, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(X_train, y_train)
    st.write('Optimal number of features :', rfecv.n_features_)
    st.write('Best features :', X_train.columns[rfecv.support_])
    fig = plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score of number of selected features")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    st.write(fig)
    


with st.spinner("Computing variance ratio..."):
    X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X_norm, y, test_size=0.25, random_state=1234)
    st.subheader('Variance ratio:')
    pca = PCA()
    pca.fit(X_norm_train)
    fig = plt.figure(1, figsize=(10, 5))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    st.write(fig)
    
'''with st.spinner("Computing parameters grid..."):
    st.warning('This may take a while since it is computing the best set of parameters for training our model.')
     # For a random forest regression model, the best parameters to consider are:
    n_estimators = [30, 40, 50, 70, 100] # Number of trees in the forest
    max_depth = [11, 15, 17, 20] # Maximum depth in a tree
    #min_samples_split = [2, 5, 7, 10] # Minimum number of data points before the sample is split
    #min_samples_leaf = [1, 5, 9, 15] # Minimum number of leaf nodes required to be sampled.
    bootstrap = [True, False] # Sampling for datapoints.
    #random_state = [42] # Generated random numbers for the random forest.
    max_features = [4,7,12,14]
    grid = get_grid_rf(n_estimators, max_depth, 0, 0, bootstrap, max_features)
    st.subheader('Parameters Grid:')
    test_scores = []
    for g in ParameterGrid(grid):
        classifier.set_params(**g) 
        classifier.fit(X_train, y_train)
        test_scores.append(classifier.score(X_test, y_test))

    best_index = np.argmax(test_scores)
    st.write(test_scores[best_index], ParameterGrid(grid)[best_index])'''



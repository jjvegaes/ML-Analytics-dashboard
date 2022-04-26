import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV
import seaborn as sns
import warnings
from bia_functions import add_params_classifier, cleaning_dataset, get_classifier, get_grid_knn, get_grid_tree, normalize, put_dataset, solve, user_input_features, get_grid_rf, get_grid_svm, plotting_metrics
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_iris, load_wine

TYPE_OF_PROBLEM = ['Classification', 
                   #'Regression'
                   ]
CLASSIFIERS = ['KNN', 'SVM', 'Random Forest', 'Decision Tree']
DATASETS = [
    #'Boston Houses', 
    'breast_cancer',
    #'Diabetes',
    'iris',
    'wine'
            ]

PLOTS = ['Confusion Matrix', 
         #'ROC Curve', # TODO: NOT WORKING ROC CURVE
         'Precision-Recall Curve','Correlation MAP', 'Variance Ratio', 'Best Features' ]

with st.spinner("Loading dataset..."):
    st.sidebar.header('User Input Parameters')
    hyper_tuning = st.sidebar.checkbox('Click here for compute brute force on hyperparameters', value=False)
    options_dataset = st.sidebar.selectbox('Select another dataset if you wish', DATASETS)
    #dataset_to_load = st.sidebar.selectbox("SELECT DATASET", DATASETS, index=0)
    X, y = put_dataset(options_dataset)
with st.spinner("Loading sidebar..."):
    classifier_name, type_of_problem, features_to_remove = user_input_features(X, TYPE_OF_PROBLEM, CLASSIFIERS)
    
with st.spinner("Cleaning and normalizing dataset..."):
    X = cleaning_dataset(X, features_to_remove)
    X_norm = normalize(X)
with st.spinner("Setting up classifier..."):
    params = add_params_classifier(classifier_name)
    classifier = get_classifier(classifier_name, params, type_of_problem)
    
with st.spinner("Training the model..."):
    #TODO: STILL HAVE TO TRY PROPHET OF FB
    X_train, X_test, y_train, y_test, y_pred = solve(X_norm, y, classifier, classifier_name)


    

with st.spinner("Feature engineering..."):
    st.header('FEATURE ENGINEERING')
    st.subheader('The dataset: ')
    st.dataframe(X)
    st.subheader('Statistics about the dataset: ')
    st.dataframe(X.describe())
    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

with st.spinner("Plotting metrics..."):
    plot_options = st.sidebar.multiselect("What to plot?", PLOTS)
    plotting_metrics(plot_options, classifier, X_test, y_test, X_norm, X_train, y_train, y, y_pred)
    
if classifier_name == 'Random Forest':
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
    

if hyper_tuning:
    with st.spinner("Computing parameters grid..."):
        st.warning('This may take a while since it is computing the best set of parameters for training our model.')
        
        if classifier_name == 'Random Forest':
            
            # For a random forest regression model, the best parameters to consider are:
            n_estimators = [20,25,30,35,40,45,50,60,70] # Number of trees in the forest
            max_depth = [8,9,10,11,13, 15, 17, 20] # Maximum depth in a tree
            #min_samples_split = [2, 5, 7, 10] # Minimum number of data points before the sample is split
            #min_samples_leaf = [1, 5, 9, 15] # Minimum number of leaf nodes required to be sampled.
            bootstrap = [True, False] # Sampling for datapoints.
            #random_state = [42] # Generated random numbers for the random forest.
            #max_features = [1]
            grid = get_grid_rf(n_estimators, max_depth, 0, 0, bootstrap, 0)
            st.subheader('Parameters Grid:')
            test_scores = []
            for g in ParameterGrid(grid):
                classifier.set_params(**g) 
                classifier.fit(X_train, y_train)
                test_scores.append(classifier.score(X_test, y_test))
            best_index = np.argmax(test_scores)
            st.write(test_scores[best_index], ParameterGrid(grid)[best_index])
            
            
        elif classifier_name == 'Decision Tree':
            
            # For a random forest regression model, the best parameters to consider are:
            max_depth = range(3,60,3) # Maximum depth in a tree
            min_samples_split = range(2,10,4) # Minimum number of data points before the sample is split
            grid = get_grid_tree(max_depth, min_samples_split)
            
            st.subheader('Parameters Grid:')
            test_scores = []
            
            for g in ParameterGrid(grid):
                classifier.set_params(**g) 
                classifier.fit(X_train, y_train)
                test_scores.append(classifier.score(X_test, y_test))
            best_index = np.argmax(test_scores)
            st.write(test_scores[best_index], ParameterGrid(grid)[best_index])
            
        elif classifier_name == 'KNN':
            K = range(1,70,25)
            ls = range(1,40,22)
            grid = get_grid_knn(K, ls)
            
            # defining parameter range
            grid = GridSearchCV(classifier, grid, cv=3, scoring='accuracy', return_train_score=False,verbose=1)
            
            # fitting the model for grid search
            grid_search=grid.fit(X_train, y_train)
            
            st.write(grid_search.best_params_)
            accuracy = grid_search.best_score_ *100
            st.write("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
            knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], leaf_size=grid_search.best_params_['leaf_size'])
            knn.fit(X, y)
            y_test_hat=knn.predict(X_test) 

            test_accuracy=accuracy_score(y_test,y_test_hat)*100

            st.write("Accuracy for our testing dataset with tuning is : {:.2f}%".format(test_accuracy) )
            fig, ax = plt.subplots()
            plot_confusion_matrix(grid,X_test, y_test,values_format='d' )
            st.pyplot(fig)
            
        elif classifier_name == 'SVM':
            
            C = [1,10]
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']
            degree = [3,10,15]
            grid = get_grid_svm(C, kernel, degree)
            
            # defining parameter range
            grid = GridSearchCV(classifier, grid, cv=3, scoring='accuracy', return_train_score=False,verbose=1)
            
            # fitting the model for grid search
            grid_search=grid.fit(X_train, y_train)
            
            st.write(grid_search.best_params_)
            accuracy = grid_search.best_score_ *100
            st.write("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )
            svm = SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'], degree=grid_search.best_params_['degree'])
            svm.fit(X, y)
            y_test_hat=svm.predict(X_test) 

            test_accuracy=accuracy_score(y_test,y_test_hat)*100

            st.write("Accuracy for our testing dataset with tuning is : {:.2f}%".format(test_accuracy) )
            fig, ax = plt.subplots()
            plot_confusion_matrix(grid,X_test, y_test,values_format='d' )
            st.pyplot(fig)
            
st.sidebar.download_button(label = 'Download dataset', data = X.to_csv(index=False), file_name='dataset.csv')



# IDEA SAVE MODELS 
'''from joblib import dump, load
>>> dump(clf, 'filename.joblib') 
>>> clf = load('filename.joblib') 

'''



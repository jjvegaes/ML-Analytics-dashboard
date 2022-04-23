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
from bia_functions import add_params_classifier, cleaning_dataset, get_classifier, get_grid_knn, get_grid_tree, normalize, put_dataset, solve, user_input_features, get_grid_rf, get_grid_svm, naive_accuracy
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

with st.spinner("Loading dataset..."):
    st.sidebar.header('User Input Parameters')
    hyper_tuning = st.sidebar.checkbox('Click here for compute brute force on hyperparameters', value=True)
    #dataset_to_load = st.sidebar.selectbox("SELECT DATASET", DATASETS, index=0)
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
    st.sidebar.download_button(label = 'Download dataset', data = X.to_csv(index=False), file_name='songs.csv')
    
with st.spinner("Training the model..."):
    X_train, X_test, y_train, y_test, y_pred = solve(X, y, classifier, classifier_name)
    

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
    select_feature = SelectKBest(f_classif, k=3).fit(X_train, y_train)
    scores = pd.concat([pd.DataFrame(data=X_train.columns),pd.DataFrame(data=select_feature.scores_[:])],axis=1)
    scores.columns = ['cat','score']
    scores = scores.sort_values('score',ascending=False)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(x='score',y='cat',data=scores, palette='seismic', ax=ax)
    plt.show()
    st.write(fig)
    
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
            

####         metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
'''
 if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maxiumum number of interations", 100, 500, key='max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?",('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classfiy", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
            
            
            
            def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix") 
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve") 
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
'''

# IDEA SAVE MODELS 
'''from joblib import dump, load
>>> dump(clf, 'filename.joblib') 
>>> clf = load('filename.joblib') 

'''
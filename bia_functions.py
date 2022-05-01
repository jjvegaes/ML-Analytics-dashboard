from ast import Global
from ensurepip import bootstrap
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import streamlit as st
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_iris, load_wine
import seaborn as sns


def normalize_data(df):
    # df on input should contain only one column with the price data (plus dataframe index)
    min = df.min()
    max = df.max()
    x = df 
    # time series normalization part
    # y will be a column in a dataframe
    y = (x - min) / (max - min)
    return y

# ADD  options_dataset in order to put other datasets, obviously
def put_dataset():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        X = pd.read_csv(uploaded_file)
    else:
        '''if options_dataset is not None:
            if options_dataset == 'iris':
                df = load_iris()
            elif options_dataset == 'breast_cancer':
                df = load_breast_cancer()
            elif options_dataset == 'wine':
                df = load_wine()
            X = pd.DataFrame(df.data, columns=df.feature_names)
            y = df.target
            return X, y
        else:'''
        X = pd.read_csv('./songs_full_data_processed.csv')
            
    X['explicit'] = X['explicit'].map({True:1, False:0}, na_action=None)
    X.dropna()
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    target_variable = st.sidebar.selectbox("Target feature", X.columns, index=len(X.columns)-1)
    y = X[target_variable].copy(deep=True)
    y = pd.factorize(y)[0]
    return X, y
    

def cleaning_dataset(dataset, features_to_remove):
    # replacing values
    for column in dataset:
        if dataset[column].dtype == 'object' and column not in features_to_remove:
            dataset[column].replace(dataset[column].unique(), range(len(dataset[column].unique())), inplace=True)
    
    for column in features_to_remove:
        try:
            dataset.drop([column], axis=1, inplace=True) 
        except:
            pass
    return dataset


def user_input_features(dataset, TYPE_OF_PROBLEM, CLASSIFIERS):
    remove = ['main_genre', 'song_id','album_id','track_number','release_date','release_date_precision','song_name','artist_id','time_signature', 'mode', 'key', 'liveness', 'valence', 'tempo', 'instrumentalness','loudness','energy','duration_ms','song_type', 'danceability', 'acousticness','popularity_song','speechiness'] #+ columns_to_remove
    st.sidebar.header("Data cleaning")
    #TODO: filtro para eliminar clases con solo una muestra, para crear un dataset distribuido
    #dataset['main_genre'].replace(dataset['main_genre'].unique(), range(len(dataset['main_genre'].unique())), inplace=True)
    #dataset = dataset.drop(dataset.groupby(by='main_genre').count() < 2, axis=0)
    #dataset.dropna()
    
    try:
        columns_to_remove = st.sidebar.multiselect("Select unnecesary features", dataset.columns, remove)
    except:
        columns_to_remove = st.sidebar.multiselect("Select unnecesary features", dataset.columns)
    st.sidebar.header("MODEL")
    classifier_name = st.sidebar.selectbox("Classifier", CLASSIFIERS, index=2)
    type_problem = st.sidebar.radio('Type of problem', TYPE_OF_PROBLEM)
    return classifier_name, type_problem, columns_to_remove


def normalize(X):
    df_model = X.copy()
    scaler = MinMaxScaler()
    #scaler = RobustScaler()
    features = [X.columns]
    for feature in features:
        try:
            df_model[feature] = scaler.fit_transform(df_model[feature])
        except:
            pass
    df_model = pd.get_dummies(df_model)
    return df_model 


def add_params_classifier(cls_name):
    params = dict()
    st.sidebar.header("Model Hyperparameters")
    if cls_name == 'KNN':
        K = st.sidebar.slider('K', 1, 70)
        leaf_size = st.sidebar.slider('Leaf size', 1, 40)
        params['K'] = K
        params['leaf_size'] = leaf_size
    elif cls_name == 'SVM':
        params['degree'] = 1
        C = st.sidebar.slider('C. Regularization parameter.', 0.01, 10.00)
        kernel = st.sidebar.selectbox("SVM KERNEL", ['linear', 'poly', 'rbf', 'sigmoid'], index = 2)
        if kernel == 'poly':
            degree = st.sidebar.slider('DEGREE POLYNOMIAL KERNEL', 1, 20)
            params['degree'] = degree
        params['C'] = C
        params['kernel'] = kernel
    elif cls_name == 'Random Forest':
        max_depth = st.sidebar.slider('Max Depth', 2, 25)
        n_estimators = st.sidebar.slider('Number of Estimators', 1, 100)
        min_samples_split = st.sidebar.slider('Minimum samples split', 0.01, 1.00)
        min_samples_leaf = st.sidebar.slider('Minimum samples leaf', 1, 15)
        #max_features = st.sidebar.slider('Maximum number of features', 1, 20)
        bootstrap = st.sidebar.checkbox('Bootstrap', value=True)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        params['min_samples_split'] = min_samples_split
        params['min_samples_leaf'] = min_samples_leaf
        params['bootstrap'] = bootstrap
        #params['max_features'] = max_features
    elif cls_name == 'Decision Tree':
        max_depth = st.sidebar.slider('Max Depth', 2, 60, step=2)
        min_samples_split = st.sidebar.slider('Minimum samples split', 2, 10)
        params['max_depth'] = max_depth
        params['min_samples_split'] = min_samples_split
    elif cls_name == 'Logistic Regression':
        C = st.sidebar.number_input("C (Regularizaion parameter)", 0.01, 10.0, key='C_LR')
        max_iter = int(st.sidebar.number_input("Maxiumum number of interations", 150, 600, key='max_iter'))
        params['C'] = C
        params['max_iter'] = max_iter
    return params


def get_classifier(cls_name, params, type_of_problem):
    if type_of_problem == 'Classification':
        if cls_name == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=params['K'], leaf_size=params['leaf_size'])
        elif cls_name == 'SVM':
            classifier = SVC(C= params['C'], kernel=params['kernel'], degree=params['degree'])
        elif cls_name == 'Random Forest':
            classifier = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42, min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], bootstrap=params['bootstrap'])
        elif cls_name == 'Decision Tree':
            classifier = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
    else:
        if cls_name == 'KNN':
            classifier = KNeighborsRegressor(n_neighbors=params['K'], leaf_size=params['leaf_size'])
        elif cls_name == 'SVM':
            classifier = SVR(C= params['C'])
        elif cls_name == 'Random Forest':
            classifier = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
        elif cls_name == 'Decision Tree':
            classifier = DecisionTreeRegressor(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
        elif cls_name == 'Logistic Regression':
            classifier = LogisticRegression(C=params['C'], max_iter=params['max_iter'])
    return classifier


def solve(df_model, y, classifier, classifier_name, TYPE_OF_PROBLEM):
    X_train, X_test, y_train, y_test = train_test_split(df_model, y, test_size=0.25, random_state=1234)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    if TYPE_OF_PROBLEM == 'Classification':
        acc = metrics.accuracy_score(y_test, y_pred) # TODO: ERROR continuous is not supported with regression, only classification here
    else:
        acc = classifier.score(X_test, y_test) # TODO: ERROR ValueError: Unknown label type: 'continuous'
    st.success(f"""
            # Classifier: {classifier_name}  #
            # Accuracy: {acc}""")
    
    pca = PCA(2)
    X_projected = pca.fit_transform(df_model)
    x1, x2 = X_projected[:, 0], X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2, c=y ,alpha=0.8, cmap='viridis')
    plt.xlabel(f'Dimensional reduction, principal component 1')
    plt.ylabel('Dimensional reduction, principal component 2')
    plt.colorbar()
    st.pyplot(fig)
    st.write("Accuracy ", acc)
    
    if classifier_name == 'Decision Tree':
        tree.plot_tree(classifier)
    return X_train, X_test, y_train, y_test, y_pred


def get_grid_rf(n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap, max_features):
    grid_rf = {
        'n_estimators': n_estimators,  
        'max_depth': max_depth,  
        #'min_samples_split': min_samples_split, 
        #'min_samples_leaf': min_samples_leaf,  
        'bootstrap': bootstrap, 
        'random_state': [42],
        #'max_features': max_features
    }
    return grid_rf


def get_grid_tree(max_depth, min_samples_split):
    grid = { 
        'max_depth': max_depth,  
        'min_samples_split': min_samples_split
    }
    return grid


def get_grid_knn(k, leaf_size):
    grid_knn = {
        'n_neighbors': k,
        'leaf_size': leaf_size
    }
    return grid_knn

def get_grid_svm(c, kernel, degree):
    grid = {
        'C': c,
        'kernel': kernel,
        'degree': degree
    }
    return grid

def naive_accuracy(true, pred):
    number_correct = 0
    i = 0
    for y in true:
        if pred[i] == y:
            number_correct += 1.0
    return number_correct / len(true)


def plotting_metrics(metrics_list, classifier, x_test, y_test, X, X_train, y_train, y, y_pred, TYPE_OF_PROBLEM):  
    #TODO: NOT WORKING ROC CURVE
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve") 
        fig, ax = plt.subplots()
        metrics.plot_roc_curve(classifier, x_test, y_test, ax=ax)
        
        # Creating visualization with the readable labels
        visualizer = metrics.roc_auc_score(y_test, y_pred, multi_class='ovo')
                     
        # Fitting to the training data first then scoring with the test data  
        # st.write('Remember: Problem np.float64 dosnt have attribute fit')
        st.dataframe(y_train)   
        visualizer.fit(X_train, y_train)                               
        visualizer.score(x_test, y_test)
        visualizer.show()
        st.write(fig)
    
    if 'Precision-Recall Curve' in metrics_list:
        try: # ONLY IN BINARY OR MULTICLASS CLASSIFICATION WE CAN APPLY THIS
            # We dont need these matrices because we already plot them
            #st.write("Precision: ", metrics.precision_score(y_test, y_pred, labels=y, average=None))
            #st.write("Recall: ", metrics.recall_score(y_test, y_pred, labels=y, average=None))
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            metrics.plot_precision_recall_curve(classifier, x_test, y_test, ax=ax)
            st.write(fig)
        except:
            pass

    if 'Correlation MAP' in metrics_list:
        with st.spinner("Correlation MAP..."):
            st.subheader('Correlation MAP: ')
            #correlation map
            f,ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)
            st.write(f)

    if 'Confusion Matrix' in metrics_list:
        try: # ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
            with st.spinner("Confusion matrix..."): # We can use this for example with the percent of change and try to classify if the day will be profit or loss
                f,ax = plt.subplots(figsize=(10, 5))
                cm = confusion_matrix(y_test,classifier.predict(x_test))
                labels = ['True Neg','False Pos','False Neg','True Pos']
                labels = np.asarray(labels).reshape(2,2)
                sns.heatmap(cm,annot=labels, fmt='', ax=ax, cmap='Blues')
                st.subheader('Confusion matrix: ')
                st.write(f)
        except:
            pass

    if 'Best Features' in metrics_list:
        with st.spinner("Finding best features..."):
            # find best scored 3 features
            st.subheader('Finding best features:')
            if TYPE_OF_PROBLEM == 'Classification':
                select_feature = SelectKBest(f_classif, k=3).fit(X_train, y_train)
            else:
                select_feature = SelectKBest(f_regression, k=3).fit(X_train, y_train)
                
            scores = pd.concat([pd.DataFrame(data=X_train.columns),pd.DataFrame(data=select_feature.scores_[:])],axis=1)
            scores.columns = ['cat','score']
            scores = scores.sort_values('score',ascending=False)
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.barplot(x='score',y='cat',data=scores, palette='seismic', ax=ax)
            plt.show()
            st.write(fig)
            #df.loc[:, df.columns != 'b']
            #df.drop('b', axis=1)
            importances = classifier.feature_importances_
            sorted_index = np.argsort(importances)[::-1]
            x_values = range(len(importances))
            labels = np.array(X.columns)[sorted_index]
            fig, ax = plt.subplots(figsize=(14, 10))
        
            plt.bar(x_values, importances[sorted_index], tick_label=labels)
            plt.xticks(rotation=90)
            plt.show()
            st.write(fig)
            
            
    if 'Variance Ratio' in metrics_list:
        with st.spinner("Computing variance ratio..."):
            X_norm_train, X_norm_test, y_norm_train, y_norm_test = train_test_split(X, y, test_size=0.25, random_state=1234)
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
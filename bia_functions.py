from ensurepip import bootstrap
from matplotlib import pyplot as plt
from sklearn import metrics
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


def normalize_data(df):
    # df on input should contain only one column with the price data (plus dataframe index)
    min = df.min()
    max = df.max()
    x = df 
    # time series normalization part
    # y will be a column in a dataframe
    y = (x - min) / (max - min)
    return y


def put_dataset():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        X = pd.read_csv(uploaded_file)
    else:
        X = pd.read_csv('./songs_full_data_processed.csv')
    X.dropna()
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    target_variable = st.sidebar.selectbox("Target feature", X.columns, index=len(X.columns)-1)
    y = X[target_variable].copy(deep=True)
    #X.drop([target_variable], axis=1, inplace=True)
    y = pd.factorize(y)[0]
    return X, y
    

def cleaning_dataset(dataset, features_to_remove):
    # replacing values
    #dataset['main_genre'].replace(dataset.main_genre.unique(), range(len(dataset.main_genre.unique())), inplace=True)
    dataset['name'].replace(dataset.name.unique(), range(len(dataset.name.unique())), inplace=True)
    dataset['song_type'].replace(dataset.song_type.unique(), range(len(dataset.song_type.unique())), inplace=True)
    dataset['artist_type'].replace(dataset.artist_type.unique(), range(len(dataset.artist_type.unique())), inplace=True)
    for column in features_to_remove:
        try:
            dataset.drop([column], axis=1, inplace=True) 
        except:
            pass
    return dataset


def user_input_features(dataset, TYPE_OF_PROBLEM, CLASSIFIERS):
    remove = ['song_id','album_id','track_number','release_date','release_date_precision','song_name','artist_id','time_signature', 'main_genre', 'mode', 'key', 'liveness', 'valence', 'tempo', 'instrumentalness','loudness','energy','duration_ms','song_type', 'danceability', 'acousticness','popularity_song']
    try:
        columns_to_remove = st.sidebar.multiselect("Select unnecesary features", dataset.columns, remove)
    except:
        columns_to_remove = st.sidebar.multiselect("Select unnecesary features", dataset.columns)
    classifier_name = st.sidebar.selectbox("Classifier", CLASSIFIERS, index=3)
    type_problem = st.sidebar.radio('Type of problem', TYPE_OF_PROBLEM)
    return classifier_name, type_problem, columns_to_remove


def normalize(X):
    df_model = X.copy()
    scaler = StandardScaler()
    #scaler = RobustScaler()

    features = [X.columns]
    for feature in features:
        df_model[feature] = scaler.fit_transform(df_model[feature])
    df_model = pd.get_dummies(df_model)
    return df_model 


def add_params_classifier(cls_name):
    params = dict()
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
    return classifier


def solve(df_model,y, classifier, classifier_name):
    #Classification
    X_train, X_test, y_train, y_test = train_test_split(df_model, y, test_size=0.25, random_state=1234)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    st.success(f"""
            # Classifier: {classifier_name}
            # Accuracy: {acc}""")
    
    pca = PCA(2)
    X_projected = pca.fit_transform(df_model)
    x1, x2 = X_projected[:, 0], X_projected[:, 1]
    fig = plt.figure()
    plt.scatter(x1, x2, c=y ,alpha=0.8, cmap='viridis')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.colorbar()
    st.pyplot(fig)
    if classifier_name == 'tree':
        #tree.plot_tree(classifier)
        pass
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
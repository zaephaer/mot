import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(layout="wide")

st.write("""
# Python Machine Learning Classification Project
Explore different dataset and classifier interactively.

Credit to (Tutorial from):
- Python Engineer (https://www.youtube.com/watch?v=Klqn--Mu2pE)
- Data Processor (https://www.youtube.com/watch?v=8M20LyCZDOY)
- Misra Turp (https://www.youtube.com/watch?v=-IM3531b1XU)
""")
#----------------------------------------------------------------------------------------------------------
st.subheader("Dataset: ")
dataset_name = st.selectbox("Selection", ("Iris","Breast Cancer","Wine"))

from sklearn import datasets
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    col = data.feature_names
    tar = data.target_names
    return X,y,col,tar
X,y,col,tar = get_dataset(dataset_name)

cold = pd.DataFrame(col)
tard = pd.DataFrame(tar)
Xd = pd.DataFrame(X, columns=cold[0])
yd = pd.DataFrame(y)
#st.header(dataset_name)
st.write("Dataset Features", X.shape,"and Target", y.shape, "shape")
st.write("Data Sample", Xd.head())
st.write("Number of Unique target", len(np.unique(y)))
st.write(tard)
st.write("Data Describe", Xd.describe())
st.write("-------------------------------------")
#----------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
#----------------------------------------------------------------------------------------------------------
scol, dcol = st.beta_columns([1,1])
scol.header("Plot Dataset")
scol.pyplot(fig)

import seaborn as sns
dcol.header("Correlation")
df2 = pd.concat([Xd, yd], axis = 1)
df2.rename({0: 'Target'}, axis=1, inplace=True)

corr = df2.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(9,7))
ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True)
dcol.pyplot(f)
#dcol.write(corr["Target"].sort_values(ascending=False))
#----------------------------------------------------------------------------------------------------------
scol, dcol = st.beta_columns([1,2])

scol.header("Classifier ")
classifier_name = scol.selectbox("Selection", ("KNN","SVM","Random Forest"))

scol.subheader("Classifier Parameter: ")
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = scol.slider("Number of K Neighbour (Default = 5)", 1, 15, 5)
        params["K"] = K
    elif clf_name == "SVM":
        C = scol.slider("Number of C (Regularization param, Default = 1.0):", 0.01, 10.0, 1.0, step = 0.5)
        params["C"] = C
    else:
        max_depth = scol.slider("Max Depth", 1, 15, 1)
        n_estimators = scol.slider("Number of Estimators (Default = 100)", 10, 100, 100, step = 10)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params
params = add_parameter_ui(classifier_name)
#----------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(max_depth = params["max_depth"],
                                    n_estimators = params["n_estimators"], random_state = 1234)
    return clf
clf = get_classifier(classifier_name, params)
#----------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
scol.subheader("Model Accuracy: ")
acc = acc * 100
scol.write(round(acc,2))
#----------------------------------------------------------------------------------------------------------
dcol.header("Prediction")
dcol.write('Specify Input (Feature Variables): ')
def user_input_features(dataset_name):
    if dataset_name == "Iris":
        sepal_length = dcol.slider('sepal_length', 4.3, 7.9, 5.8)
        sepal_width = dcol.slider('sepal_width', 2.0, 4.4, 3.0)
        petal_length = dcol.slider('petal_length', 1.0, 6.9, 3.7)
        petal_width = dcol.slider('petal_width', 0.1, 2.5, 1.2)
        features = {'sepal_length':sepal_length, 'sepal_width':sepal_width,
                    'petal_length':petal_length, 'petal_width':petal_width}
    elif dataset_name == "Breast Cancer":
        mean_radius = dcol.slider('mean_radius', 6.98, 28.11, 14.12)
        mean_texture = dcol.slider('mean_texture', 9.71, 39.28, 19.28)
        mean_perimeter = dcol.slider('mean_perimeter', 43.79, 188.50, 91.96)
        mean_area = dcol.slider('mean_area', 143.50, 2501.00, 654.88)
        mean_smoothness = dcol.slider('mean_smoothness', 0.05, 0.16, 0.09) #5
        mean_compactness = dcol.slider('mean_compactness', 0.01, 0.34, 0.10)
        mean_concavity = dcol.slider('mean_concavity', 0.00, 0.41, 0.08)
        mean_concave_points = dcol.slider('mean_concave_points', 0.00, 0.20, 0.04)
        mean_symmetry = dcol.slider('mean_symmetry', 0.10, 0.30, 0.18)
        mean_fractal_dim = dcol.slider('mean_fractal_dim', 0.05, 0.09, 0.06) #10
        radius_error = dcol.slider('radius_error', 0.11, 2.87, 0.40)
        texture_error = dcol.slider('texture_error', 0.36, 4.88, 1.21)
        perimeter_error = dcol.slider('perimeter_error', 0.75, 21.98, 2.86)
        area_error = dcol.slider('area_error', 6.80, 542.20, 40.33)
        smoothness_error = dcol.slider('smoothness_error', 0.00, 0.03, 0.01) #15
        compactness_error = dcol.slider('compactness_error', 0.00, 0.13, 0.02)
        concavity_error = dcol.slider('concavity_error', 0.00, 0.39, 0.03)
        concave_point_error = dcol.slider('concave_point_error', 0.00, 0.05, 0.01)
        symmetry_error = dcol.slider('symmetry_error', 0.00, 0.07, 0.02)
        fractal_dimension_error = dcol.slider('fractal_dimension_error',0.0009, 0.0298, 0.0038) #20
        worst_radius = dcol.slider('worst_radius', 7.93, 36.04, 16.26)
        worst_texture = dcol.slider('worst_texture', 12.02, 49.54, 25.67)
        worst_perimeter = dcol.slider('worst_perimeter', 50.41, 251.20, 107.26)
        worst_area = dcol.slider('worst_area', 185.20, 4254.00, 880.58)
        worst_smoothness = dcol.slider('worst_smoothness', 0.07, 0.22, 0.13) #25
        worst_compactness = dcol.slider('worst_compactness', 0.02, 1.05, 0.25)
        worst_concavity = dcol.slider('worst_concavity', 0.00, 1.25, 0.27)
        worst_concave_points = dcol.slider('worst_concave_points', 0.00, 0.29, 0.11)
        worst_symmetry = dcol.slider('worst_symmetry', 0.15, 0.66, 0.29)
        worst_fractal_dimension = dcol.slider('worst_fractal_dimension', 0.05, 0.20, 0.08) #30
        features = {'mean_radius':mean_radius, 'mean_texture':mean_texture,'mean_perimeter':mean_perimeter, 'mean_area':mean_area,'mean_smoothness':mean_smoothness,
                    'mean_compactness':mean_compactness, 'mean_concavity':mean_concavity, 'mean_concave_points':mean_concave_points, 'mean_symmetry':mean_symmetry, 'mean_fractal_dim':mean_fractal_dim,
                    'radius_error':radius_error, 'texture_error':texture_error, 'perimeter_error':perimeter_error, 'area_error':area_error, 'smoothness_error':smoothness_error,
                    'compactness_error':compactness_error, 'concavity_error':concavity_error, 'concave_point_error':concave_point_error, 'symmetry_error':symmetry_error, 'fractal_dimension_error':fractal_dimension_error,
                    'worst_radius':worst_radius, 'worst_texture':worst_texture, 'worst_perimeter':worst_perimeter, 'worst_area':worst_area, 'worst_smoothness':worst_smoothness,
                    'worst_compactness':worst_compactness, 'worst_concavity':worst_concavity, 'worst_concave_points':worst_concave_points, 'worst_symmetry':worst_symmetry, 'worst_fractal_dimension':worst_fractal_dimension}
    else:
        alcohol = dcol.slider('alcohol', 11.03, 14.83, 13.00)
        malic_acid = dcol.slider('malic_acid', 0.74, 5.80, 2.33)
        ash = dcol.slider('ash', 1.36, 3.23, 2.36)
        alcalinity_of_ash = dcol.slider('alcalinity_of_ash', 10.60, 30.00, 19.49)
        magnesium = dcol.slider('magnesium', 70.00, 162.00, 99.74) #5
        total_phenols = dcol.slider('total_phenols', 0.98, 3.88, 2.29)
        flavanoids = dcol.slider('flavanoids', 0.34, 5.08, 2.02)
        nonflavanoid_phenols = dcol.slider('nonflavanoid_phenols', 0.13, 0.66, 0.36)
        proanthocyanins = dcol.slider('proanthocyanins', 0.41, 3.58, 1.59)
        color_intensity = dcol.slider('color_intensity', 1.28, 13.00, 5.05) #10
        hue = dcol.slider('hue', 0.48, 1.71, 0.95)
        od280_od315_of_diluted_wines = dcol.slider('od280_od315_of_diluted_wines', 1.27, 4.00, 2.61)
        proline = dcol.slider('proline', 278.00, 1680.00, 746.89)
        features = {'alcohol':alcohol, 'malic_acid':malic_acid,'ash':ash, 'alcalinity_of_ash':alcalinity_of_ash,
                    'magnesium':magnesium, 'total_phenols':total_phenols, 'flavanoids':flavanoids, 'nonflavanoid_phenols':nonflavanoid_phenols,
                    'proanthocyanins':proanthocyanins, 'color_intensity':color_intensity, 'hue':hue, 'od280_od315_of_diluted_wines':od280_od315_of_diluted_wines,
                    'proline':proline }
    df_features = pd.DataFrame(features, index=[0])
    return df_features
df_features = user_input_features(dataset_name)
#----------------------------------------------------------------------------------------------------------
dcol.write('Selected Input parameters')
dcol.write(df_features)
dcol.write("Result represent:")
dcol.write(tard)
dcol.write("Prediction RESULT:")
dcol.write(int(clf.predict(df_features)))

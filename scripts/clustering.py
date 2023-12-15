from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
import inspect
import numpy as np, pandas as pd

def cluster_purity(y_true, y_pred):
    confomat = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-confomat)
    return confomat[row_ind, col_ind].sum() / np.sum(confomat)

def redux_fit(model, components: int, randstate: int, data: pd, **kwargs) -> pd:
    np.random.seed(randstate)
    model_args = inspect.signature(model).parameters
    if 'method' in model_args:
        method = kwargs.pop('method', None)
        if method:
            redux = model(n_components=components, method=method).fit_transform(data)
        else:
            redux = model(n_components=components).fit_transform(data)
    else:
        redux = model(n_components=components).fit_transform(data)
    X=redux[:, 0]
    y=redux[:, 1]
    new_df = pd.DataFrame([X, y]).transpose()
    new_df.rename(columns={0:'X', 1:'y'}, inplace=True)
    return new_df
    
def redux_transform(model, components: int, randstate: int, data: pd, **kwargs):
    np.random.seed(randstate)
    model_args = inspect.signature(model).parameters
    if 'method' in model_args:
        method = kwargs.pop('method', None)
        if method:
            redux = model(n_components=components, method=method).fit_transform(data)
        else:
            redux = model(n_components=components).fit_transform(data)
    else:
        redux = model(n_components=components).fit_transform(data)
    return redux

def kmeans_redux(n_clusters: list[int], model, components: int, randstate: int, data: pd, y_true, **kwargs):
    if type(y_true[0]) != int:
        encoding = LabelEncoder()
        numeric = encoding.fit_transform(y_true)
    cluster_purities = []
    redux = redux_transform(model, components, randstate, data)
    
    for cluster in n_clusters:
        kmeans = KMeans(n_clusters=cluster, random_state=42, n_init=10)
        kmeans.fit(redux)
        labels = kmeans.labels_
        purity = cluster_purity(y_true=numeric, y_pred=labels)
        cluster_purities.append(purity)
    
    return cluster_purities
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import inspect

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
  
def cluster_purity(y_true, y_pred):
    confomat = confusion_matrix(y_true, y_pred)
    # We use the Linear Assignment Problem approach to solve label switching problem.
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix_)
    return confomat[row_ind, col_ind].sum() / np.sum(confomat)
    

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class RenamingTransformer(BaseEstimator, TransformerMixin):
  #your __init__ method below
  def __init__(self, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
  #write the transform method without asserts. Again, maybe copy and paste from MappingTransformer and fix up.

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    #print(self.mapping_dict)
    X_ = X.copy()
    X_ = X_.rename(columns = self.mapping_dict)
    #print(X.columns, X_.columns)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  #fill in the rest below
  
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    #print(self.mapping_dict)
    X_ = X.copy()
    X_ = pd.get_dummies(X_,
                        prefix = self.target_column,
                        prefix_sep = '_',
                        columns = [self.target_column],
                        dummy_na = self.dummy_na,
                        drop_first = self.drop_first    #Yeah I borroed your code from above
                        )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    #print(self.mapping_dict)
    X_ = X.copy()
    X_ = X_[self.column_list] if self.action == 'keep' else X_.drop(columns = self.column_list)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold

  #define methods below

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    #print(self.mapping_dict)
    X_ = X.copy()
    drop_table = np.triu(X_.corr(method='pearson').abs().ge(self.threshold), 1)
    drop_cols = set([X_.columns[i] for i in np.where(drop_table)[1]])
    
    X_ = X_.drop(columns=drop_cols)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def compute_3sigma_boundaries(df, column_name):
    assert isinstance(df, pd.core.frame.DataFrame), f'expected Dataframe but got {type(df)} instead.'
    assert column_name in df.columns.to_list(), f'unknown column {column_name}'
    assert all([isinstance(v, (int, float)) for v in df[column_name].to_list()])

    #compute mean of column - look for method
    m = df[column_name].mean()
    #compute std of column - look for method
    sigma = df[column_name].std()
    return  m - 3 * sigma, m + 3 * sigma #(lower bound, upper bound)

  def transform(self, X):
    #print(self.mapping_dict)
    X_ = X.copy()

    minb, maxb = compute_3sigma_boundaries(X_, self.target_column)
    X_[self.target_column] = X_[self.target_column].clip(lower=minb, upper=maxb)

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence
  
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    q1 = X_[self.target_column].quantile(0.25)
    q3 = X_[self.target_column].quantile(0.75)
    iqr = q3-q1
    outer_low = q1-3*iqr
    outer_high = q3+3*iqr
    inner_low = q1 - 1.5*iqr
    inner_high = q3 + 1.5*iqr
    if self.fence == 'inner':
      X_[self.target_column] = X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    elif self.fence == 'outer':
      X_[self.target_column] = X_[self.target_column].clip(lower=outer_low, upper=outer_high)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  #fill in rest below
  def fit(self, X, y = None):
    print("not implemented")
    return X

  def transform(self, mtx):
    new_df = mtx.copy()
    for column in new_df:
      mi = new_df[column].min()
      mx = new_df[column].max()
      denominator = mx - mi
      new_df[column] -= mi #x - min(x) <- top of function
      new_df[column] /= denominator
    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

  
class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  #your code
  def fit(self, X, y = None):
    print(f"Warning: {self.__class__.__name__}.fit does nothing.")
    return self

  def transform(self, X):
    X_ = X.copy()
    knn = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=self.add_indicator)
    return pd.DataFrame(knn.fit_transform(X_), columns=X_.columns)

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

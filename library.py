import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
model = LogisticRegressionCV(random_state=1, max_iter=5000)

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
  
  
def find_random_state(df, labels, n=200):
# idx = np.array(abs(var - rs_value)).argmin()
  errors = []
  for i in range(1, n):
    x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)  # predict against training set
    x_test_pred = model.predict(x_test)  # predict against test set
    train_error = f1_score(y_train, x_train_pred)  # how bad did we do with prediction on training data?
    test_error = f1_score(y_test, x_test_pred) # how bad did we do with prediction on test data?
    errors.append(test_error / train_error) # take the ratio
  
  rs_value = sum(errors)/len(errors)
  return np.array(abs(errors - rs_value)).argmin()


# titanic transformer
titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)


# customer transformer
customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)


def dataset_setup(feature_table, labels, the_transformer, rs=1234, ts=.2):
  X_train, X_test, y_train, y_test = train_test_split(feature_table, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)

  x_trained_numpy = X_train_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  x_test_numpy = X_test_transformed.to_numpy()
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=88, ts=.2):
  return dataset_setup(titanic_table.drop(columns='Survived'), titanic_table['Survived'].to_list(), transformer, rs, ts)


def customer_setup(customer_table, transformer=customer_transformer, rs=107, ts=.2):
  return dataset_setup(customer_table.drop(columns='Rating'), customer_table['Rating'].to_list(), transformer, rs, ts)


def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(
      columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >= t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {
        'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
  return result_df


def halving_search(model, grid, x_train, y_train, factor=3, scoring='roc_auc'):
  halving_cv = HalvingGridSearchCV(
      model, grid,  # our model and the parameter combos we want to try
      scoring=scoring,  # could alternatively choose f1, accuracy or others
      n_jobs=-1,
      min_resources="exhaust",
      factor=factor,  # a typical place to start so triple samples and take top 3rd of combos on each iteration
      cv=5, random_state=1234,
      refit=True  # remembers the best combo and gives us back that model already trained and ready for testing
  )

  return halving_cv.fit(x_train, y_train)


#New additions to library

class SplittingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, target_column, split_names:list, split_target:str, drop=True):  
    self.split_names = split_names      #New column names
    self.split_target = split_target    #Where to split on cell value
    self.target_column = target_column  #column to focus on
    self.drop = drop                    #Remove the old column

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    X_[self.split_names] = X_[self.target_column].str.split(self.split_target, expand=True)

    if self.drop:
      del X_[self.target_column]

    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class CombiningTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, new_column, target_columns:list):  
    self.new_column = new_column            #New column name
    self.target_columns = target_columns    #Which columns to add together

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    
    X_[self.new_column] = X_[self.target_columns].sum(axis=1)
      
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class NaNsTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, target_columns:list, value):  
    self.target_columns = target_columns  
    self.value = value

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    X_[self.target_columns].replace(to_replace=np.nan, value=self.value)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

#final transformer  
enterprise_transformer = Pipeline(steps=[
    ('Home', OHETransformer(target_column='HomePlanet')),
    ('Cryo', MappingTransformer('CryoSleep', {False: 0, True: 1})),
    ('Cabin', SplittingTransformer('Cabin', ['Deck', 'Num', 'Side'], '/')),
    ('Dest', OHETransformer(target_column='Destination')),
    ('Age', TukeyTransformer('Age', 'outer')),
    ('VIP', MappingTransformer('VIP', {False: 0, True: 1})),
    ('NaNs', NaNsTransformer(['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'], 0.0)),
    ('New', CombiningTransformer('TotalSpent', ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])),
    ('Room', TukeyTransformer('RoomService', 'outer')),
    ('Food', TukeyTransformer('FoodCourt', 'outer')),
    ('Shop', TukeyTransformer('ShoppingMall', 'outer')),
    ('Spa', TukeyTransformer('Spa', 'outer')),
    ('VR', TukeyTransformer('VRDeck', 'outer')),
    ('Total', TukeyTransformer('TotalSpent', 'outer')),
    ('Deck', OHETransformer(target_column='Deck')),
    ('Side', MappingTransformer('Side', {'P': 0, 'S': 1})),
    ('Drop', DropColumnsTransformer(['PassengerId', 'Name', 'Num'], 'drop')),
    ('scale', MinMaxTransformer()), 
    ('imputer', KNNTransformer())
    ], verbose=True)

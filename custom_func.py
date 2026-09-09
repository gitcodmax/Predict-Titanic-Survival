import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def load_datasets():
  X_train = pd.read_csv(r'..\proc_data\X_train.csv')
  X_test = pd.read_csv(r'..\proc_data\X_test.csv')
  y_train = pd.read_csv(r'..\proc_data\y_train.csv')

  return X_train, X_test, y_train

def sub_file(test_preds, filename):
  sub_df = pd.read_csv(r'..\gender_submission.csv')
  sub_df['Survived'] = test_preds
  sub_df.to_csv(f'..\\submissions\\{filename}.csv', index=None)
  print(sub_df)

def split_evaluate(model, X_train, y_train):

  kf = KFold(n_splits=3, shuffle=True, random_state=42)

  train_as = []
  val_as = []

  for train_idx, val_idx in kf.split(X_train, y_train):
    print(len(train_idx), len(val_idx))
    train_input = X_train.iloc[train_idx]
    val_input = X_train.iloc[val_idx]

    train_target = y_train.iloc[train_idx]
    val_target = y_train.iloc[val_idx]

    model.fit(train_input, train_target)

    train_preds = model.predict(train_input)
    val_preds = model.predict(val_input)

    train_as.append(accuracy_score(train_target, train_preds))
    val_as.append(accuracy_score(val_target, val_preds))

  print(f'Training Score: {np.mean(train_as)}')
  print(f'Validation Score: {np.mean(val_as)}')
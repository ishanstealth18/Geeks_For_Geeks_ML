import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.indexers import objects
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('Zillow.csv')

# Remove columns with 1 value and columns with more than 60% mean NaN values
remove_col = []

for col in df.columns:
    if df[col].nunique() == 1:
        remove_col.append(col)
    elif (df[col].isna()).mean() > 0.60:
        remove_col.append(col)

print(len(remove_col))
df.drop(remove_col, axis=1, inplace=True)
print(df.info())

# Impute NaN values
nan_cols = []

for col in df.columns:
    if df[col].isna().sum() > 0:
        nan_cols.append(col)

#print(nan_cols)

fig, axes = plt.subplots(4,8, figsize=(15, 8))
flat_axs = axes.flat

for i,col in enumerate (nan_cols):
    if df[col].dtype != 'object':
        ax = flat_axs[i]
        ax.hist(df[col], bins=25, color='rebeccapurple', edgecolor='white', alpha=0.8)
        ax.set_title(f'{col}')
        ax.set_ylabel('Count')

plt.suptitle('DataFrame Columns Histograms', fontsize=16)
plt.tight_layout()
plt.show()

# From the distribution, we can see that most of the columns distribution is either left or right skewed
# this means we can use median values to impute NaN values for numeric columns and mode for category

for col in nan_cols:
    if df[col].dtype == 'object':
        df.fillna({col: df[col].mode()[0]}, inplace=True)
    else:
        df.fillna({col: df[col].median()}, inplace=True)

# check outliers
df.drop(['propertycountylandusecode', 'propertyzoningdesc'], axis=1, inplace=True)


fig1, axes1 = plt.subplots(4,7, figsize=(15, 8))
flat_axs = axes1.flat

for i,col in enumerate (df.columns):
    ax1 = flat_axs[i]
    ax1.boxplot(df[col], notch=False, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='navy'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='orange', markersize=6))
    ax1.set_title(f'{col}')
    ax1.set_ylabel('Count')

plt.suptitle('Boxplot', fontsize=16)
plt.tight_layout()
plt.show()


ints, objects, floats = [], [], []

for col in df.columns:
    if df[col].dtype == float:
        floats.append(col)
    elif df[col].dtype == int:
        ints.append(col)
    else:
        objects.append(col)

len(ints), len(floats), len(objects)

print('Shape of the dataframe before removal of outliers', df.shape)
df = df[(df['target'] > -1) & (df['target'] < 1)]
print('Shape of the dataframe after removal of outliers ', df.shape)

import seaborn as sns

for col in objects:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr() > 0.8,
           annot=True,
           cbar=False)
plt.show()

to_remove = ['calculatedbathnbr', 'fullbathcnt', 'fips',
             'rawcensustractandblock', 'taxvaluedollarcnt',
             'finishedsquarefeet12', 'landtaxvaluedollarcnt']

df.drop(to_remove, axis=1, inplace=True)

features = df.drop(['parcelid'], axis=1)
target = df['target'].values

X_train, X_val,\
    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.1,
                                      random_state=22)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from sklearn.metrics import mean_absolute_error as mae
models = [LinearRegression(),
          Lasso(), RandomForestRegressor(), Ridge()]

for i in range(4):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')

    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    val_preds = models[i].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()
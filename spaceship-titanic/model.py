
# -- importing libraries --

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# -- Load a dataset into a Pandas Dataframe -- (8693, 14)
dataset_df = pd.read_csv('data/train.csv')
print("Full train dataset shape is {}".format(dataset_df.shape))

# -- Display the first 5 examples -- 
#print(dataset_df.head(5))

#print("\n", dataset_df.describe())

#dataset_df.info()

# -- Bar chart for the transported column --

plot_df = dataset_df.Transported.value_counts()
plot_df.plot(kind="bar")

fig, ax = plt.subplots(5,1,  figsize=(10, 10))
plt.subplots_adjust(top = 2)

# -- Numerical data distribution

sns.histplot(dataset_df['Age'], color='b', bins=50, ax=ax[0]);
sns.histplot(dataset_df['FoodCourt'], color='b', bins=50, ax=ax[1]);
sns.histplot(dataset_df['ShoppingMall'], color='b', bins=50, ax=ax[2]);
sns.histplot(dataset_df['Spa'], color='b', bins=50, ax=ax[3]);
sns.histplot(dataset_df['VRDeck'], color='b', bins=50, ax=ax[4]);


# -- Non necessary columns (name, ID) --
dataset_df = dataset_df.drop(['PassengerId', 'Name'], axis=1)
print(dataset_df.head(5))

# print(dataset_df.isnull().sum().sort_values(ascending=False))

# -- Filling NA with zeros --
dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = dataset_df[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)

print(dataset_df.isnull().sum().sort_values(ascending=False))


# -- Boolean => int (as boolean not supported by TFDF)
label = "Transported"
dataset_df[label] = dataset_df[label].astype(int)

dataset_df['VIP'] = dataset_df['VIP'].astype(int)
dataset_df['CryoSleep'] = dataset_df['CryoSleep'].astype(int)

# -- splitting Cabin column , then removing it
dataset_df[["Deck", "Cabin_num", "Side"]] = dataset_df["Cabin"].str.split("/", expand=True)

try:
    dataset_df = dataset_df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

print(dataset_df.head(5))
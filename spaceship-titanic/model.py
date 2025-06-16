
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

# -- Splitting --
def split_dataset(dataset, test_ratio=0.20):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

# -- Converting to TF Datasets --
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)



rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=["accuracy"]) # setting metric to be evaluated 

# -- Training --
rf.fit(x=train_ds)

# colab = tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)


# -- Evaluation --
evaluation = rf.evaluate(x=valid_ds,return_dict=True)

for name, value in evaluation.items():
  print(f"{name}: {value:.4f}")

inspector = rf.make_inspector()
inspector.variable_importances()["NUM_AS_ROOT"]

# SUBMISSION
# Load the test dataset
test_df = pd.read_csv('data/test.csv')
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to 1's and 0's
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert pd dataframe to tf dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Get the predictions for testdata
predictions = rf.predict(test_ds)

n_predictions = (predictions > 0.5).astype(bool)

output = pd.DataFrame({'PassengerId': submission_id,
                       'Transported': n_predictions.squeeze()})

# print(output)

sample_submission_df = pd.read_csv('sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('submission.csv', index=False)
sample_submission_df.head()
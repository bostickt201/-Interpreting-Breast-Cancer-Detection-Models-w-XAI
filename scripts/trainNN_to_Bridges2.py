# imports
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_csv('/Users/taylor/Desktop/DS340W/term_project/data/wisconsin/wisconsin_train_balanced.csv')
dataset = df.values

# split into input (X) and output (Y) variables
X = dataset[:,0:30].astype(float)
Y = dataset[:,30]

# encode class values as integers
encoder = LabelEncoder()

encoder.fit(Y)
encoded_Y = encoder.transform(Y)

##############################################################################

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(30, input_shape=(30,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=32, verbose=0) # baseline epochs chosen by 3 * ncol
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

##############################################################################

# Repeat w/ smaller model:

def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(15, input_shape=(30,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_smaller, epochs=100, batch_size=32, verbose=0)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

##############################################################################

# Repeat w/ larger model (i.e., 10, ..., 13 hidden layers):

# larger model
def create_larger():
    # create model
    model = Sequential()

    model.add(Dense(30, input_shape=(30,), activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_larger, epochs=100, batch_size=32, verbose=0)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)

print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

##############################################################################

# Evaluate deep (12-hidden layers) nn performance on test data:

test_df = pd.read_csv('/Users/taylor/Desktop/DS340W/term_project/data/wisconsin/wisconsin_test.csv')
test_dataset = test_df.values

test_X = test_dataset[:,0:30].astype(float)
test_Y = test_dataset[:,30]

encoder.fit(test_Y)
encoded_test_Y = encoder.transform(test_Y)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_larger, epochs=100, batch_size=32, verbose=0)))

pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, test_X, encoded_test_Y, cv=kfold)

print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

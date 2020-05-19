""" Simple linear topology neural network """
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

FINAL_TEST_ROWS = 3000

df = pandas.read_csv("behavior.csv", usecols=lambda x: x != '')
isMalware = df['tag'].values
del df['tag']  # We should not use the answer as input data
df.drop(df.columns[0], axis=1, inplace=True)  # 0th column is some kind of hash
xLen = len(df.columns)
df.reindex(np.random.permutation(df.index))
inputs = df.values

X = inputs.astype(float)
Y = isMalware


def createModel():
    model = Sequential()
    model.add(Dense(xLen, input_dim=xLen, activation='relu'))
    model.add(Dense(xLen // 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


classifier = KerasClassifier(build_fn=createModel, epochs=30, batch_size=32, verbose=2, validation_split=0.2)
classifier.fit(X[FINAL_TEST_ROWS:], Y[FINAL_TEST_ROWS:])

res = classifier.predict(X[:FINAL_TEST_ROWS])
np.set_printoptions(threshold=np.inf)  # Print full arrays
print(np.transpose(res)[0], Y[:FINAL_TEST_ROWS], sep='\n')

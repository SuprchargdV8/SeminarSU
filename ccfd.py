# NECESSARY PACKAGES

import os
import numpy as np
import pandas as pd

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score

# keras NN
from keras import layers
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential 
from keras import regularizers

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras import metrics


#--------------------#
#  RANDOM FUNCTIONS  #
#--------------------#

def dimensionality_plot(X, y):
    sns.set(style='whitegrid', palette='muted')
    # Initializing TSNE object with 2 principal components
    tsne = TSNE(n_components=2, random_state = 42)
    
    # Fitting the data
    X_trans = tsne.fit_transform(X)
    
    plt.figure(figsize=(12,8))
    
    plt.scatter(X_trans[np.where(y == 0), 0], X_trans[np.where(y==0), 1], marker='o', color='g', linewidth=1, alpha=0.8, label='Normal')
    plt.scatter(X_trans[np.where(y == 1), 0], X_trans[np.where(y==1), 1], marker='o', color='k', linewidth=1, alpha=0.8, label='Fraud')
    
    plt.legend(loc = 'best')
    
    plt.show()


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()


# LOADING AND PREPEARING THE DATA
data = pd.read_csv("creditcard.csv")

print(data.head())
print(data.shape)
print(data.describe())

class_names = {0:'Not Fraud', 1:'Fraud'}
print(data.Class.value_counts().rename(index = class_names))
data.groupby('Class')['Class'].count().plot.bar(logy=True)
fig = plt.figure(figsize = (15, 12))

data['Time'] = data['Time'].apply(lambda t: (t/3600) % 24 )

plt.subplot(5, 6, 1) ; plt.plot(data.V1) ; plt.subplot(5, 6, 15) ; plt.plot(data.V15)
plt.subplot(5, 6, 2) ; plt.plot(data.V2) ; plt.subplot(5, 6, 16) ; plt.plot(data.V16)
plt.subplot(5, 6, 3) ; plt.plot(data.V3) ; plt.subplot(5, 6, 17) ; plt.plot(data.V17)
plt.subplot(5, 6, 4) ; plt.plot(data.V4) ; plt.subplot(5, 6, 18) ; plt.plot(data.V18)
plt.subplot(5, 6, 5) ; plt.plot(data.V5) ; plt.subplot(5, 6, 19) ; plt.plot(data.V19)
plt.subplot(5, 6, 6) ; plt.plot(data.V6) ; plt.subplot(5, 6, 20) ; plt.plot(data.V20)
plt.subplot(5, 6, 7) ; plt.plot(data.V7) ; plt.subplot(5, 6, 21) ; plt.plot(data.V21)
plt.subplot(5, 6, 8) ; plt.plot(data.V8) ; plt.subplot(5, 6, 22) ; plt.plot(data.V22)
plt.subplot(5, 6, 9) ; plt.plot(data.V9) ; plt.subplot(5, 6, 23) ; plt.plot(data.V23)
plt.subplot(5, 6, 10) ; plt.plot(data.V10) ; plt.subplot(5, 6, 24) ; plt.plot(data.V24)
plt.subplot(5, 6, 11) ; plt.plot(data.V11) ; plt.subplot(5, 6, 25) ; plt.plot(data.V25)
plt.subplot(5, 6, 12) ; plt.plot(data.V12) ; plt.subplot(5, 6, 26) ; plt.plot(data.V26)
plt.subplot(5, 6, 13) ; plt.plot(data.V13) ; plt.subplot(5, 6, 27) ; plt.plot(data.V27)
plt.subplot(5, 6, 14) ; plt.plot(data.V14) ; plt.subplot(5, 6, 28) ; plt.plot(data.V28)
plt.subplot(5, 6, 29) ; plt.plot(data.Amount)
plt.show()

feature_names = data.iloc[:, 1:30].columns
target = data.iloc[:1, 30: ].columns
print(feature_names)
print(target)

data_features = data[feature_names]
data_target = data[target]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)
print("Length of X_train is:", len(X_train))
print("Length of X_test is:", len(X_test))
print("Length of y_train is:", len(y_train))
print("Length of y_test is:", len(y_test))

# BASIC LOGISTIC REGRESSION

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())

pred = model.predict(X_test)

class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

f1_scoreC = round(f1_score(y_test, pred), 2)
recall_score = round(recall_score(y_test, pred), 2)
auc_score = round(roc_auc_score(y_test, pred), 2)
print("Report of BLR: \n", classification_report(y_test, pred))

# AUTOENCODERS

# reduce set
normal_trans = data[data['Class'] == 0].sample(200000)
fraud_trans = data[data['Class'] == 1] 
reduced_set = normal_trans.append(fraud_trans).reset_index(drop=True)

# split into features
y = reduced_set['Class']
X = reduced_set.drop('Class', axis=1)

print(f"Shape of Features : {X.shape} and Target: {y.shape}")

#dimensionality_plot(X, y)

scaler = RobustScaler().fit_transform(X)

# Scaled data
X_scaled_normal = scaler[y == 0]
X_scaled_fraud = scaler[y == 1]

print(f"Shape of the input data : {X.shape[1]}")

# Input layer with a shape of features/columns of the dataset
input_layer = Input(shape = (X.shape[1], ))

# Construct encoder network
encoded = Dense(100, activation= 'tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(25, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(12, activation = 'tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(6, activation='relu')(encoded)

# Decoder network
decoded = Dense(12, activation='tanh')(encoded)
decoded = Dense(25, activation='tanh')(decoded)
decoded = Dense(50, activation='tanh')(decoded)
decoded = Dense(100, activation='tanh')(decoded)

output_layer = Dense(X.shape[1], activation='relu')(decoded)

# Building a model
auto_encoder = Model(input_layer, output_layer)

# Compile the auto encoder model
auto_encoder.compile(optimizer='adadelta', loss='mse')

# Training the auto encoder model
auto_encoder.fit(X_scaled_normal, X_scaled_normal, batch_size=32, epochs=20, shuffle=True, validation_split=0.20)

latent_model = Sequential()
latent_model.add(auto_encoder.layers[0])
latent_model.add(auto_encoder.layers[1])
latent_model.add(auto_encoder.layers[2])
latent_model.add(auto_encoder.layers[3])
latent_model.add(auto_encoder.layers[4])

# Predictions
normal_tran_points = latent_model.predict(X_scaled_normal)
fraud_tran_points = latent_model.predict(X_scaled_fraud)

# Making as a one collection
encoded_X = np.append(normal_tran_points, fraud_tran_points, axis=0)
y_normal = np.zeros(normal_tran_points.shape[0])
y_fraud = np.ones(fraud_tran_points.shape[0])
encoded_y = np.append(y_normal, y_fraud, axis=0)

# Calling TSNE plot function
dimensionality_plot(encoded_X, encoded_y)

X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(encoded_X, encoded_y, test_size=0.3)

print(f"Encoded train data X: {X_enc_train.shape}, Y: {y_enc_train.shape}, X_test :{X_enc_test.shape}, Y_test: {y_enc_test.shape}")
print(f"Actual train & test data X: {X_train.shape}, Y: {y_train.shape}, X_test :{X_test.shape}, Y_test: {y_test.shape}")

# Instance of SVM
svc_clf = SVC()

svc_clf.fit(X_train, y_train)
svc_predictions = svc_clf.predict(X_test)

print("Report of AE:\n ", classification_report(y_test, svc_predictions))


# Decision trees
tempTr_X,  tempTe_X, tempTr_y, tempTe_y = train_test_split(X_train, y_train, test_size=0.3)
scores = []
spec = []
sens = []
for i in np.arange(2,10):
    classifier = DecisionTreeClassifier(max_depth=i)
    classifier.fit(tempTr_X, tempTr_y)
    predicted = classifier.predict(tempTe_X)
    scores.append(f1_score(tempTe_y, predicted))
    temp = classification_report(tempTe_y, predicted, output_dict=True)
    spec.append(temp["0"]["recall"])
    sens.append(temp["1"]["recall"])

plt.title("Decision trees")
plt.xlabel("depth")
plt.ylabel("score")
plt.plot(np.arange(2,10), scores, "r", label="F1 score")
plt.plot(np.arange(2,10), spec, "g", label="Specificity")
plt.plot(np.arange(2,10), sens, "b", label="Sensitivity")
plt.legend()
plt.show()

classifier = DecisionTreeClassifier(max_depth=(np.arange(2,10)[np.argmax(scores)]))
classifier.fit(X_test, y_test)
predicted = classifier.predict(X_test)
print("Report of DT:\n", classification_report(y_test, predicted))
# >>> F1 score of DT: 0.929


# KNN
tempTr_X,  tempTe_X, tempTr_y, tempTe_y = train_test_split(X_train, y_train, test_size=0.3)
scores = []
spec = []
sens = []
for i in np.arange(2,10):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(tempTr_X, tempTr_y)
    predicted = classifier.predict(tempTe_X)
    scores.append(f1_score(tempTe_y, predicted))
    temp = classification_report(tempTe_y, predicted, output_dict=True)
    spec.append(temp["0"]["recall"])
    sens.append(temp["1"]["recall"])

plt.title("KNN")
plt.xlabel("number of neighbors")
plt.ylabel("score")
plt.plot(np.arange(2,10), scores, "r", label="F1 score")
plt.plot(np.arange(2,10), spec, "g", label="Specificity")
plt.plot(np.arange(2,10), sens, "b", label="Sensitivity")
plt.legend()
plt.show()

classifier = KNeighborsClassifier(n_neighbors=(np.arange(2,10)[np.argmax(scores)]))
classifier.fit(X_test, y_test)
predicted = classifier.predict(X_test)
print("Report of KNN:\n", classification_report(y_test, predicted))
# >>>  F1 score : 0.790

# Random Forest
tempTr_X,  tempTe_X, tempTr_y, tempTe_y = train_test_split(X_train, y_train, test_size=0.3)
scores = []
spec = []
sens = []
for i in np.arange(50,150,10):
    classifier = RandomForestClassifier(n_estimators=i)
    classifier.fit(tempTr_X, tempTr_y)
    predicted = classifier.predict(tempTe_X)
    scores.append(f1_score(tempTe_y, predicted))
    temp = classification_report(tempTe_y, predicted, output_dict=True)
    spec.append(temp["0"]["recall"])
    sens.append(temp["1"]["recall"])

plt.title("Random forest")
plt.xlabel("number of trees")
plt.ylabel("score")
plt.plot(np.arange(50,150,10), scores, "r", label="F1 score")
plt.plot(np.arange(50,150,10), spec, "g", label="Specificity")
plt.plot(np.arange(50,150,10), sens, "b", label="Sensitivity")
plt.legend()
plt.show()

classifier = RandomForestClassifier(n_estimators=(np.arange(50,150,10)[np.argmax(scores)]))
classifier.fit(X_test, y_test)
predicted = classifier.predict(X_test)
print("Report of RF:\n", classification_report(y_test, predicted))
# >>>  F1 score:  0.997


# PREPEARING THE DATA FOR NN

sampleDF = data.iloc[:,:]

shuffleDF = shuffle(sampleDF, random_state=42)

df_train = shuffleDF[0:250000]
df_test = shuffleDF[250000:]

train_feature = np.array(df_train.values[:,0:29])
train_label = np.array(df_train.values[:,-1])
test_feature = np.array(df_test.values[:,0:29])
test_label = np.array(df_test.values[:,-1])

scaler = MinMaxScaler()

scaler.fit(train_feature)
train_feature_trans = scaler.transform(train_feature)
test_feature_trans = scaler.transform(test_feature)

# KERAS NN

model = Sequential()
model.add(Dense(units=200, input_dim=29, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

train_history = model.fit(x=train_feature_trans, y=train_label, validation_split=0.8, epochs=200, batch_size=500, verbose=2)

show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(test_feature_trans, test_label)
print('\n')
print('accuracy=',scores[1])


predicted = model.predict(test_feature_trans)

print("Report: \n",classification_report(test_label,predicted.flatten().round()))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv(r'C:\Users\Mark\Documents\844\A1\adult.data', header=None)

data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'martial-status', 'occupation', 'relationship', 'race', 'sex', 
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
                'class']

data2 = data[:]

categorical_columns = ['workclass', 'education', 'martial-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country', 'class']

#Convert categorical variables into numerical format
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

drop_class = ['class', 'workclass', 'occupation', 'education', 'education-num'] # for analysis

X = data.drop(drop_class, axis=1)  # Features
Y = data['class']  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# Initialize the classifiers
svm_classifier = SVC()
tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5)
knn_classifier = KNeighborsClassifier(n_neighbors=100)
nb_classifier = GaussianNB()
ann_model_classifier = tf.keras.Sequential([ # Artificial Neural Network model
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
ann_model_classifier.compile(optimizer='adam', # ANN model need to be compiled
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the classifiers
svm_classifier.fit(X_train, y_train)
tree_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
nb_classifier.fit(X_train, y_train)
ann_model_classifier.fit(X_train, y_train)


# Predict the labels for the test set using the 5 different classifiers
y_pred_SVM = svm_classifier.predict(X_test)
y_pred_Decision_tree = tree_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)
test_loss, accuracy5 = ann_model_classifier.evaluate(X_test, y_test)



# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_SVM)
print("Accuracy using SVM:", accuracy)

accuracy2 = accuracy_score(y_test, y_pred_Decision_tree)
print("Accuracy using Decision Tree:", accuracy2)

accuracy3 = accuracy_score(y_test, y_pred_knn)
print("Accuracy using K-Nearest Neighbors:", accuracy3)

accuracy4 = accuracy_score(y_test, y_pred_nb)
print("Accuracy using Naive Bayes:", accuracy4)

print("Accuracy using Artificial Neural Network:", accuracy5)


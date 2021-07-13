import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import unittest
from sklearn.model_selection import cross_val_score

# # Reading the data

df = pd.read_csv('include/winequality-red.csv', delimiter=';')

df.head()


# # Initial visual analysis

for label in df.columns[:-1]:
#for label in ['alcohol']:
	plt.scatter(df['quality'], df[label]) 
	plt.title(label)
	plt.xlabel('quality')
	plt.ylabel(label)
	plt.savefig('imgs/'+'red'.join(label.split(' ')))
#	plt.show()

# # Gathering the training and testing data

# Since the numbers 3-9 don't really mean much, lets map these to low(0), mid(1), and high(2)
bins = [0, 5.5, 7.5, 10] # this means 3-5 are low, 6-7 are mid, 8-9 are high
labels = [0, 1, 2]
df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)
X = df.drop(columns=['quality'])

# df.head()
x = df[df.columns[:-1]]
y = df['quality']
sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)


for data in [y_train, y_test]:
    print(data.describe())


# # K Nearest Neighbors Classifier
print("\nK Nearest Neighbors Classifier\n")

print("3 Neighbors\n")

#def data_knn3(self):
n3 = KNeighborsClassifier(n_neighbors = 3)
n3.fit(x_train, y_train)
pred_n3 = n3.predict(x_test)
print("Classification Report\n")
print(classification_report(y_test, pred_n3, labels=np.unique(pred_n3)))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_n3))
confusion_matrix(y_test, pred_n3)

##Accuracy
print('Accuracy of Knn 3 :')
actual_n3 = accuracy_score(y_test, pred_n3)
print(actual_n3)
#return actual_n3

	##KNN3 unittest
	# class Test_KNN3(unittest.TestCase):
	# 	def test_knn3(self):
	# 		expected_n3 = 65 <= actual_n3 <= 70
	# 		assert actual_n3, expected_n3

print("\n5 Neighbors\n")

#def data_knn5(self):
n5 = KNeighborsClassifier(n_neighbors = 5)
n5.fit(x_train, y_train)
pred_n5 = n5.predict(x_test)
print("Classification Report\n")
print(classification_report(y_test, pred_n5, labels=np.unique(pred_n5)))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_n5))
##Accuracy
print('Accuracy of Knn 5 :')
actual_n5 = accuracy_score(y_test, pred_n5)
print(actual_n5)
#return actual_n5
	##KNN5 unittest
	# class Test_KNN5(unittest.TestCase):
	# 	def test_knn5(self):
	# 		expected_n5 = 65 <= actual_n5 <= 70
	# 		assert actual_n5, expected_n5

##########Cross Validation of KNN

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
print('\ncv scores : ')
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

# from sklearn.model_selection import cross_val_score,cross_val_predict
# k_range = range(1, 31)
# # list of scores from k_range
# k_scores = []
# #loop through reasonable values of k
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     #obtain cross_val_score for KNNClassifier with k neighbours
#     scores = cross_val_score(knn, X_scaled, wine.target, cv=5, scoring='accuracy')
#     #append mean of scores for k neighbors to k_scores list
#     k_scores.append(scores.mean())
# print(k_scores)
# Image for post
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')

#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

# #K-Means Clustering Algorithm
print('\nK-Means Clustering Algorithm\n')

kmeans = KMeans(n_clusters=4)
km = kmeans.fit(np.reshape(x,(len(x),11)))
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
print(labels)

pred_km = km.predict(x_test)
print("Classification Report\n")
print(classification_report(y_test, pred_km, labels=np.unique(pred_km), zero_division=1))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_km))

##Accuracy
print('Accuracy of K-Means :')
print(accuracy_score(y_test, pred_km))

colors = ["g.","r.","y.","b."]

for i in range(len(x)):
	plt.plot(x[i], colors[labels[i]], markersize = 10)
	centroid_x = centroids[:,0]
	centroid_y = centroids[:,1]
	
# plt.scatter(centroid_x,centroid_y, marker = "x", s = 150, linewidths = 5, zorder = 10)
# plt.show()


#}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

# # Random Forest Classifier

print("\nRandom Forest Classifiera\n")

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
print("Classification Report\n")
print(classification_report(y_test, pred_rf, labels=np.unique(pred_rf), zero_division=1))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_rf))

# # Decision Tree Classifier

print("\nDecision Tree Classifier\n")

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
pred_dt = dt.predict(x_test)
print(classification_report(y_test, pred_dt))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_dt))


# # Stochastic Gradient Descent

print("\nStochastic Gradient Descent\n")

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
pred_sgd = sgd.predict(x_test)
print("Classification Report\n")
print(classification_report(y_test, pred_sgd,labels=np.unique(pred_sgd)))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred_sgd))


# # Trying to improve results

# number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
# # number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # max number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=10)] + [None]
# # min number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # min number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # method of selecting samples for training each tree
# bootstrap = [True, False]

# random grid
# random_grid = {'n_estimators': n_estimators,
#               'max_features': max_features,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               'bootstrap': bootstrap}

# rf_optimized = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42)

# print(rf_optimized.best_params_)

# rf_optimized.fit(x_train, y_train)
# pred_optimized = rf_optimized.predict(x_test)
# print(classification_report(y_test, pred_optimized))


# print(classification_report(y_test, pred_optimized))
# print(rf_optimized.best_params_)
# if __name__ == '__main__':
#     unittest.main()
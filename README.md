# imageprocessing for begginers 
 let's learn the basics of image processing which involves image acquisition , image preprocessing , and classifying the image using the following algorithms knn, random forest classifier, decision tree, naive bayes
 I have used google collab and cifar10 dataset 
 ### using random forest 
 ```
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import numpy as np
import cv2
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data() #The dataset can be loaded using the code 
x_train.shape,x_test.shape
# Normalization
x_train = x_train/255.0
x_test = x_test/255.0
#sklearn expects i/p to be 2d array-model.fit(x_train,y_train)=>reshape to 2d array
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))
#so,eventually,model.predict() should also be a 2d input
nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples,nx*ny*nrgb))

#For this, we must first import it from sklearn:
from sklearn.ensemble import RandomForestClassifier
#Create an instance of the RandomForestClassifier class:
model=RandomForestClassifier()
#Finally, let us proceed to train the model:
model.fit(x_train2,y_train)
#Now, predict for the test set using the fitted Random Forest Classifier model:

y_pred=model.predict(x_test2)
y_pred
# Now, evaluate the model with the test images by obtaining its classification report, confusion matrix, and accuracy score.

accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))
confusion_matrix(y_pred,y_test)
# Option 1: Raw string with 'r' prefix
img_path = '/bird.jfif'



img_arr = cv2.imread(img_path)
img_arr = cv2.resize(img_arr, (32, 32))
#Now, reshape the image to 2D as discussed in the pre-processing section:

#so,eventually,model.predict() should also be a 2d input
nx, ny, nrgb = img_arr.shape
img_arr2 = img_arr.reshape(1,(nx*ny*nrgb))
#Let us declare a list called classes:

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
ans=model.predict(img_arr2)
print(classes[ans[0]])
#RandomForestClassifier
```
### using knn 
```
from sklearn.neighbors import KNeighborsClassifier
#and then instantiating it to create a KNN model:

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train2,y_train)
#Now, predict for the test set using the fitted KNN model:

y_pred_knn=knn.predict(x_test2)
y_pred_knn
accuracy_score(y_pred_knn,y_test)
print(classification_report(y_pred_knn,y_test))
confusion_matrix(y_pred_knn,y_test)
```
### using decision tree
```
from sklearn.tree import DecisionTreeClassifier
#and then instantiating it to create a DecisionTreeClassifier model:

dtc=DecisionTreeClassifier()
#Finally, train it:

dtc.fit(x_train2,y_train)
#Now, predict for the test set using the fitted decision tree model:

y_pred_dtc=dtc.predict(x_test2)
y_pred_dtc
accuracy_score(y_pred_dtc,y_test)
print(classification_report(y_pred_dtc,y_test))
confusion_matrix(y_pred_dtc,y_test)
```
### naive  bayes
```
from sklearn.naive_bayes import GaussianNB
#and then instantiating it to create an NB model:

nb=GaussianNB()
#Finally, train it:

nb.fit(x_train2,y_train)
#Now, predict for the test set using the fitted NB model:

y_pred_nb=nb.predict(x_test2)
y_pred_nb
accuracy_score(y_pred_nb,y_test)
print(classification_report(y_pred_nb,y_test))
confusion_matrix(y_pred_nb,y_test)
```

# Example (churning customers) 😰

#Import standard Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import modeling Librarry
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

# Import function that display the splits of the tree
from sklearn.tree import plot_tree


#Read in data
file = "../Datasets/Churn_Modeling.csv 
data = pd.read.csv(file)
data.head()

#here you do Exploratory Data Analysis (EDA), decide appropriate evaluation metrics

#Check class balance
data['Exited'].value_counts()

#Now we calculate average balance of customers who churned (Exit from service)
average_churned_balance = data[data['Exited']==1]['Balance'].mean()
average_churned_balance

#Now we're prepararing our data for modeling

#We have to create new data new variable 'churn_data' that drops value such as RowNumber, Surname and Gender, CustomerId
churn_data = data.drop['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1)
churn_data.head()

#Now we have to create boolean column (ENCODE) using pandas library
churn_data = pd.get_dummies(churn_data, drop_first=True)
churn_data.head()

#We have to split our data into test and train 
y = churn_data['Exited'] #Our goal column

x = chirn_data.copy()
x = x.drop('Exited', axis=1)

x_train, x_test, y_train, y_test = train_test(x, y, test_size=0.25, stratify=y, random_state=40) #We use train_test_split function, Stratify (To avoid bias, more reliable evaluation)



#Train base line decision model
#Instantiate the Model
tree_decision = DecisionTreeClassifier(random_state=0)

#Now we have to fit the model to training data
tree_decision.fit(X_train, y_train)

#Then we making predictions on test data (predict method)
data_pred = tree_decision.predict(X_test)

#Now we generate performance metrics
print('Accuracy':, "%.3f" % accuracy_score(y_test, data_pred))
print('Precision':, "%.3f" % precision_score(y_test, data_pred))
print('Recall':, "%.3f" % recall_score(y_test, data_pred))
print('F1 Score':, "%.3f" % f1_score(y_test, data_pred))

# Inspect Confusion Matrix plot
def confusion_matrix_plot(model, x_data, y_data):
  '''Returns a plot of confusion matrix for predictions on y data'''
  
  model_pred = model_predict(x_data)
  cm = confussion_matrix(y_daya, model_pred, label=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_label=model.classes_)

  disp.plot()
  plt.show()

# Generate Confusion matrix
confusion_matrix_plot(tree_decision, x_test, y_test)


# Finally, plot the tree
plt.figure(figsize=(15, 12))
plot_tree(tree_decision, max_depth=2, fontsize=14, feature_names=x.columns, class_names={0: 'stayed', 1:'churned'}, filled=True); # If we didn't set the max_depth the function will return entire tree,class_name (what the majority class of each node is), filled(color the nodes according to the majority class)
plt.show()

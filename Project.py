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

x = churn_data.copy()
x = x.drop('Exited', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=40) #We use train_test_split function, Stratify (To avoid bias, more reliable evaluation)


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
  cm = confussion_matrix(y_data, model_pred, label=model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_label=model.classes_)
  disp.plot()
  plt.show()

# Generate Confusion matrix
confusion_matrix_plot(tree_decision, x_test, y_test)

# Finally, plot the tree
plt.figure(figsize=(15, 12))
plot_tree(tree_decision, max_depth=2, fontsize=14, feature_names=x.columns, class_names={0: 'stayed', 1:'churned'}, filled=True); # If we didn't set the max_depth the function will return entire tree,class_name (what the majority class of each node is), filled(color the nodes according to the majority class)
plt.show()

# Tune and validate decision trees with Python

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate the classifier 
tuned_decision_tree = DecisionTreeClassifier(random_state=42)

# Assign a dictionary of hyperparameters to search over
tree_paramateres = {'max_depth': [4,5,6,7,8,9,10,12,15,20,30,40,50], 'min_samples_leaf': [2, 5, 10, 20, 50]}

# Assign a set of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object. Pass as arguments: The classifier (tuned_decision_tree), The dictionary of hyperparameters to search over (tree_para), The set of scoring metrics (scoring),The number of cross-validation folds you want (cv=5), The scoring metric that you want GridSearch to use when it selects the "best" model (i.e., the model that performs best on average over all validation folds) (refit='f1'*)
%%time
clf = GridSearchCV(tuned_decision_tree, tree_parameters, scoring=scoring, cv=5, refit="f1")

# Fit the data (X_train, y_train) to the GridSearchCV object (clf)
clf.fit(x_train, y_train)

# Examine the best model for GridSearch, we can use the (best_estimator_)
clf.best_estimator_

# Now, print yours results
print("Best Avg. Validation Score: ", "%.4f" % clf.best_score_)


# These other metrics are much more directly interpretable, so they're worth knowing. The following cell defines a helper function that extracts these scores from the fit GridSearchCV object and returns a pandas dataframe with all four scores from the model with the best average F1 score during validation. This function will help us later when we want to add the results of other models to the table.
def make_results(model_name, model_object):
    
    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)
​
    # Isolate the row of the df with the max(mean f1 score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]
​
    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                         }
                        )
  
    return table
    
# Call above function to our model
result_table = make_results('Tuned Decision Tree', clf)

# Save the table as csv (we can save then these results and open them in another notebook, use to.csv() from pandas library)
result_table.to_csv('Results.csv)

# Check the results
result_table
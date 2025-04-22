import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender  # Updated import
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve  # Correct import for learning_curve
from sklearn import metrics
from sklearn.preprocessing import StandardScaler  # Updated import
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # Updated import

####### function for reading dataset from csv files

def read_datasets():
    """ Reads users profile from csv files """ 
    genuine_users = pd.read_csv("C:/Users/srija/Downloads/Fake-profile/Fake-Profile-Detection/data/fusers.csv")
    fake_users = pd.read_csv("C:/Users/srija/Downloads/Fake-profile/Fake-Profile-Detection/data/users.csv")
    x = pd.concat([genuine_users, fake_users])
    y = len(fake_users) * [0] + len(genuine_users) * [1]
    return x, y

####### function for predicting sex using name of person

def predict_sex(name):
    sex_predictor = gender.Detector(case_sensitive=False)  # Updated gender detector
    first_name = name.str.split(' ').str.get(0)
    sex = first_name.apply(sex_predictor.get_gender)
    
    # Mapping output to numerical codes
    sex_dict = {
        'female': -2,
        'mostly_female': -1,
        'unknown': 0,
        'andy': 0,  # 'andy' means androgynous in gender-guesser
        'mostly_male': 1,
        'male': 2
    }
    
    sex_code = sex.map(sex_dict).astype(int)
    return sex_code

####### function for feature engineering

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'])))   
    lang_dict = { name: i for i, name in lang_list }             
    x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)
    x.loc[:, 'sex_code'] = predict_sex(x['name'])
    feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'sex_code', 'lang_code']
    x = x.loc[:, feature_columns_to_use]
    return x

####### function for plotting learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

####### function for plotting confusion matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['Fake', 'Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####### function for plotting ROC curve

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    print("False Positive rate: ", false_positive_rate)
    print("True Positive rate: ", true_positive_rate)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

####### Function for training data using Random Forest

def train(X_train, y_train, X_test):
    """ Trains and predicts dataset with a Random Forest classifier """
    
    clf = RandomForestClassifier(n_estimators=40, oob_score=True)
    clf.fit(X_train, y_train)
    print("The best classifier is: ", clf)
    
    # Estimate score
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(scores)
    print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
    
    title = 'Learning Curves (Random Forest)'
    plot_learning_curve(clf, title, X_train, y_train, cv=5)
    plt.show()
    
    # Predict 
    y_pred = clf.predict(X_test)
    return y_test, y_pred

print("reading datasets.....\n")
x, y = read_datasets()
x.describe()

print("extracting features.....\n")
x = extract_features(x)
print(x.columns)
print(x.describe())

print("splitting datasets in train and test dataset...\n")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

print("training datasets.......\n")
y_test, y_pred = train(X_train, y_train, X_test)

import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import sys
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)

#Read in the data
df_1 = pd.read_csv('C:\\Users\\jpkli\\Desktop\\Machine Learning Projects\\Mental Health\\survey.csv')
df_2 = pd.read_csv('C:\\Users\\jpkli\\Desktop\\Machine Learning Projects\\Mental Health\\'
                   'mental-heath-in-tech-2016_20161114.csv')


#Comments section, while informative, is not going to be looked at for this data.
df_1 = df_1.drop(['comments'], axis = 1)
df_1 = df_1.drop(['Timestamp'], axis = 1)
#print(df_1.info())


#Age column had some irregular values, so values were replaced with the mean.
#To do this, we made values nan values, then replaced nan values with mean.

df_1.loc[df_1['Age']<15] = np.nan
df_1.loc[df_1['Age']>75] = np.nan

df_1['Age'].fillna((df_1['Age'].mean()), inplace=True)
df_1['Age']=df_1['Age'].round(0)

#print(df_1['Age'].unique())


#Here we need to fix the gender column.
#First we take any nan values and replace them with ''.
df_1['Gender'] = df_1['Gender'].fillna('')


#Now we make everything lowercase and work on replacing all values to either male, female or other for simplicity's sake.
gender_lower = df_1['Gender'].str.lower()
gender = gender_lower.unique()

male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
            "cis man", "Cis Male", "cis male"]

other_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
             "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman",
             "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means",
             "A little about you", "p", ""]

female_str = ["cis female", "f", "female", "woman",  "femake", "female ", "cis-female/femme", "female (cis)",
              "femail"]

for (row, col) in df_1.iterrows():

    if str.lower(col.Gender) in male_str:
        df_1['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        df_1['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in other_str:
        df_1['Gender'].replace(to_replace=col.Gender, value='other', inplace=True)


#print(df_1['Gender'].unique())


#Changing nan country and state values to other. Most likely will drop columns later.
df_1['Country'] = df_1['Country'].fillna('other')
#print(df_1['Country'].unique())

df_1['state'] = df_1['state'].fillna('other')
#print(df_1['state'].unique())


#In checking to see how many nan values are in the self employed column, there are very few missing data, and
#since the column is prominently 'no' we will replace missing data with 'no'.

#print(Counter(df_1['self_employed']))
df_1['self_employed'] = df_1['self_employed'].fillna('No')
#print(Counter(df_1['self_employed']))


#Same deal for family history.
#print(df_1['family_history'].unique())
#print(Counter(df_1['family_history']))
df_1['family_history'] = df_1['family_history'].fillna('No')


#This is one where I can come back to tweak later. 8 missing values, slightly more 'Yes' than 'No'.
#So for now, just make it 'Yes'.
#print(df_1['treatment'].unique())
#print(Counter(df_1['treatment']))
df_1['treatment'] = df_1['treatment'].fillna('Yes')


#For work interfere, for nan, will change to 'unsure'
#print(df_1['work_interfere'].unique())
#print(Counter(df_1['work_interfere']))
df_1['work_interfere'] = df_1['work_interfere'].fillna('Unsure')


#This is one where I can come back to tweak later. 8 missing values.
#So for now, just make it '6-25'.
#print(df_1['no_employees'].unique())
#print(Counter(df_1['no_employees']))
df_1['no_employees'] = df_1['no_employees'].fillna('6-25')


#remote_work column, showing different ways to clean data.
#print(df_1['remote_work'].unique())
#print(Counter(df_1['remote_work']))
df_1['remote_work'].fillna(df_1['remote_work'].mode()[0], inplace=True)


#To shorten the amount of time deealing with the data for now, we will use the mode to finish cleaning up the last 15 columns
cols = ['tech_company','benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
        'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

df_1[cols]=df_1[cols].fillna(df_1.mode().iloc[0])



labelDict = {}
for feature in df_1:
    le = preprocessing.LabelEncoder()
    le.fit(df_1[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    df_1[feature] = le.transform(df_1[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] = labelValue

for key, value in labelDict.items():
    print(key, value)

# Get rid of 'Country'
df_1 = df_1.drop(['Country'], axis=1)
print(df_1.head())


MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost
    xgb.XGBClassifier()
]


#splitting data into training data and test data
cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.7, random_state=0)


#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean',
               'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

#create table to compare MLA predictions
df_x_bin = cols
MLA_predict = df_1['benefits']

print(len(df_1['treatment']))

#indexing through MLA and saving the performance to a table
row_index = 0
for alg in MLA:
    # set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, df_1[df_x_bin], df_1['treatment'], cv=cv_split, return_train_score = True)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
                                                                'test_score'].std() * 3  # let's know the worst that can happen!

    # save MLA predictions
    alg.fit(df_1[df_x_bin], df_1['treatment'])
    MLA_predict[MLA_name] = alg.predict(df_1[df_x_bin])

    row_index += 1

#print and sort table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)


#barplot
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#make it look fancy using pyplot
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

plt.show()
print(MLA_compare)


X = df_1.drop(['treatment'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, df_1['treatment'], test_size=0.33, random_state=7)


xgb1 = XGBClassifier(
 learning_rate =0.2,
 n_estimators=250,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)


xgb1.fit(X_train, y_train)
# make predictions for test data
y_pred = xgb1.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print(xgb1.feature_importances_)
plt.bar(range(len(xgb1.feature_importances_)), xgb1.feature_importances_)
plt.show()

























#!/usr/bin/env python
# coding: utf-8

# # Importing Required libraries

# In[318]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[265]:


#Load the data
df = pd.read_csv('Downloads/Data/train_ctrUa4K.csv')
test = pd.read_csv('Downloads/Data/test_lAUu6dG.csv')
df.head()


# In[266]:


test.head()


# In[267]:


df.shape, test.shape   # There is no Target Feature in Test because we need to predict using Train data.


# In[268]:


df.describe()


# In[269]:


df.info()


# In[270]:


df['Credit_History'].value_counts()


# In[271]:


df['Loan_Status'].value_counts(normalize=True).plot(kind='bar')  


# # Fill Missing Values on Train Data.

# In[272]:


df.isnull().sum()


# In[273]:


#We will first fill missing values using mean() in numeric features.
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

#Now, we will fill missing values by taking mode() in categorical features.

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[274]:


test.info()


# In[275]:


# Univariate Analysis (Means We are Visualizing Individual Categorical Features (Independent))

plt.subplot(221)
df['Gender'].value_counts(normalize =True).plot(figsize = [15,10],kind='bar',title = "Gender")
plt.subplot(222)
df['Married'].value_counts(normalize = True).plot(kind='bar',title = "Married")
plt.subplot(223)
df['Self_Employed'].value_counts(normalize=True).plot(kind = 'bar',title = 'Self_Employed')
plt.subplot(224)
df['Education'].value_counts(normalize = True).plot(kind = 'bar',title = 'Education')


# In[276]:


# Ordinal Features (Independent)
plt.subplot(121)
df['Dependents'].value_counts(normalize=True).plot(figsize=[15,5],kind='bar',title = "Dependents")

plt.subplot(122)
df['Property_Area'].value_counts(normalize = True).plot(kind = 'bar',title = 'Property_Area')


# # Fill Missing Values on Test Data

# In[277]:


test.isnull().sum()


# In[278]:


# first fill missing values in numeric features using mean()
test['LoanAmount'] = test['LoanAmount'].fillna(test['LoanAmount'].mean())
test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean())

# Now, fill missing values in category features using mode()
test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0])
test['Self_Employed'] = test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])
test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mode()[0])


# In[279]:


df['Dependents'] = df['Dependents'].apply(lambda x : x.replace('+',''))
test['Dependents'] = test['Dependents'].apply(lambda x : x.replace('+',''))


# In[280]:


plt.figure(figsize=[15,5])
plt.subplot(121)
sns.distplot(df['LoanAmount'],bins=20)
plt.title('Train Data')
plt.subplot(122)
sns.distplot(test['LoanAmount'],bins=20)
plt.title('Test Data')


# In[281]:


# The Graph looks Right Skewed so We use log function in order to make approx Normal Distribution.
plt.figure(figsize=[15,5])
plt.subplot(121)
df['LoanAmount'] = np.log(df['LoanAmount'])
sns.distplot(df['LoanAmount'],bins=20)
plt.title('Train Data')
plt.subplot(122)
test['LoanAmount'] = np.log(test['LoanAmount'])
sns.distplot(test['LoanAmount'],bins=20)
plt.title('Test Data')


# In[282]:


plt.subplot(121)
df['ApplicantIncome'].plot(figsize=[10,5],kind='box')
plt.subplot(122)
sns.distplot(df['ApplicantIncome'],bins=20)


# We see so many Outliers are present in the ApplicantIncome.
# 
# 

# In[283]:


plt.subplot(121)
df['CoapplicantIncome'].plot(figsize = [10,5],kind='box')
plt.subplot(122)
sns.distplot(df['CoapplicantIncome'],bins=20)


# So,here we can see many outliers in the CoapplicantIncome as well.

# # Bivariate analysis (Independent Vs Dependent Features)

# So, in this type of analysis we'll see how many Independent features are affecting Dependent feature.

# In[284]:


married = pd.crosstab(df['Married'],df['Loan_Status'])
married.div(married.sum(1).astype(float),axis=0).plot(figsize=[5,5],kind='bar',stacked=True)
plt.xticks(rotation=360)
plt.xlabel('Married')
plt.ylabel('Percentage')


# In[285]:


edu = pd.crosstab(df['Education'],df['Loan_Status'])
edu.div(edu.sum(1).astype(float),axis=0).plot(figsize=[5,5],kind='bar',stacked=True)
plt.xticks(rotation=360)
plt.xlabel('Education')
plt.ylabel('Percentage')


# In[286]:


credit = pd.crosstab(df['Credit_History'],df['Loan_Status'])
credit.div(credit.sum(1).astype(float),axis=0).plot(figsize=[5,5],kind='bar',stacked=True)
plt.xticks(rotation=360)
plt.xlabel('Credit History')
plt.ylabel('Percentage')


# Credit History looks highly impacted on Dependent Feature.

# In[287]:


area = pd.crosstab(df['Property_Area'],df['Loan_Status'])
area.div(area.sum(1).astype(float),axis=0).plot(figsize=[5,5],kind='bar',stacked=True)
plt.xticks(rotation=360)
plt.xlabel('\nProperty Area')
plt.ylabel('Percentage')


# Now, let's make group of ApplicatIncome.

# In[288]:


bins = [0,3000,6000,9000,81000]
group = ['Low','Average','High','Very High']
df['Income_Applicant'] = pd.cut(df['ApplicantIncome'],bins,labels=group)
df.head()


# In[289]:


bins = [0,1000,4000,42000]
group = ['Low','High','Very High']
df['Income_Coapplicant'] = pd.cut(df['CoapplicantIncome'],bins,labels = group)
df.head()


# In[290]:


income_1 = pd.crosstab(df['Income_Applicant'],df['Loan_Status'])
income_1.div(income_1.sum(1).astype(float),axis=0).plot(figsize=[10,5],kind='bar',stacked=True)
plt.xlabel('Income of Applicant')
plt.ylabel('Percentage')


# In[291]:


income_2 = pd.crosstab(df['Income_Coapplicant'],df['Loan_Status'])
income_2.div(income_2.sum(1).astype(float),axis=0).plot(kind='bar',stacked = True)
plt.xlabel('Income of Coapplicant')
plt.ylabel('Percentage')


# In[292]:


df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.head()


# In[293]:


bins = [0,3000,8000,25000,81000]
group = ['Low','Average','High','very High']
df['Total_applicant'] = pd.cut(df['Total_Income'],bins,labels=group)
df.head()


# In[294]:


total = pd.crosstab(df['Total_applicant'],df['Loan_Status'])
total.div(total.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('Total Income')
plt.ylabel('Percentage')


# Let's remove the features which we created for exploration analysis and we note that if the loan amount is low 
# then the chances of approval Loan would be High.

# In[295]:


df = df.drop(['Income_Applicant','Income_Coapplicant','Total_Income','Total_applicant'],axis=1)
df.head()


# In[296]:


plt.figure(figsize=[9,8])
sns.heatmap(df.corr(),square=True,cmap='YlGnBu',annot=True)


# Before Moving to Build Machine Learning Algorithm Drop Loan_Id Feature, which is no effect on target feature.

# In[297]:


df = df.drop(['Loan_ID'],axis=1)
test = test.drop(['Loan_ID'],axis=1)
test.head()


# Now drop Target feature from train data and save it in a new variable called y.

# In[298]:


X = df.drop(['Loan_Status'],axis=1)
y = df['Loan_Status']


# As Model takes only Numerical Values we will convert all Category features into numerical by using pandas get_dummies function

# In[300]:


X = pd.get_dummies(X)
df = pd.get_dummies(df)
test = pd.get_dummies(test)
X.head()


# Here,we are taking X_nd instead of X_test, because we have test data so it is better to take another name just to keep things simple and we'll not do misunderstanding.

# In[301]:


X_train,X_nd,y_train,y_nd = train_test_split(X,y,test_size = 0.3,random_state = 0)


# So, In the dataset after doing dummies function we see many features contains only 0 or 1. Therefore it is good to go with Logistic regression algorithm and see how much score we are getting.

# In[303]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[305]:


nd_pred = lr.predict(X_nd)
accuracy_score(y_nd,nd_pred)


# In[306]:


dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[308]:


nd_predict1 = dt.predict(X_nd)
accuracy_score(y_nd,nd_predict1)


# In[309]:


gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)


# In[310]:


nd_pred2 = gb.predict(X_nd)
accuracy_score(y_nd,nd_pred2)


# So, we have seen that among 3 Classifier models we got 82% Accuracy to Predict Loan_Status in Logistic Regression.
# 
# Now, lets examine our model with confusion matrix.

# In[317]:


cm = confusion_matrix(y_nd,nd_pred)
print(cm)

sns.heatmap(cm,annot=True,cmap='RdBu')
plt.title('Confusion_matrix')
plt.xlabel('Predicted')
plt.ylabel('True Value')


# In[319]:


cl = classification_report(y_nd,nd_pred)
print(cl)


# Now, let's predict the test dataset and see how much score we are getting.

# In[325]:


prediction = lr.predict(test)
prediction[:50]


# Import the Sample file to do Submission.

# In[326]:


sample = pd.read_csv('Downloads/sample_submission_49d68Cx.csv')


# In[327]:


sample['Loan_Status'] = prediction 


# In[329]:


sample.to_csv('Downloads/Logistic regression.csv',index=False)


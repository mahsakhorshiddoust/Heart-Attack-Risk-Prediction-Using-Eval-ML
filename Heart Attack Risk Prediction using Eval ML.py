#!/usr/bin/env python
# coding: utf-8

# # Heart Attack Risk Prediction Using Eval ML
# 
# Mahsa Mohammadkhorshiddoust
# 

# In[1]:


#Loading The Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:\\Users\\akbar\\Desktop\\Heart-Attack\\heart-data.csv", encoding = 'ANSI', engine='python')
df.head()


# In[3]:


df = df.drop(['oldpeak', 'slp', 'thall'], axis = 1)
df.shape


# In[4]:


df.isnull().sum()


# As there are no null values in our dataset, we can go on analysing the data.

# Here are the correlation between features which indicates how are features are correlated.

# In[5]:


correlation = df.corr()
correlation


# Lets now visualize this correlation to dive deep into the analysis!

# In[6]:


sns.set(style="darkgrid", context="talk", font_scale=0.8)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (12,8))

sns.heatmap(correlation, linewidths=.1, annot=True, annot_kws={"size":9}, cmap= "Blues")
print("As we can see, negative values indicate the lighter colors, while the darker colores represent the strong relationship between features.")


# In[7]:


sns.set(style="darkgrid", context="talk", font_scale=0.8)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (15,8))

plt.title("Age of Patients")
plt.xlabel("Age")
sns.countplot(x= 'age', data = df, palette="Blues")

print("This plot shows that between the ages of 54-58, the risk of heart attack is at it's highest.")


# In[8]:


sns.set(style="darkgrid", context="talk", font_scale=0.8)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (8,3))

plt.title("Sex of Patients: Female=0, Male=1")
plt.xlabel("Sex")
sns.countplot(x= 'sex', data = df, palette="Blues", alpha=0.8)

print("This plot shows that men carry the higher risk of heart attack.")


# In[9]:


print("Now lets see the frequency of unique chest pain values by creating a new dataframe and renaming the indexes to the types of chest pains.")
ChestPain = df['cp'].value_counts().reset_index()
ChestPain['index'][3] = 'Asymptomatic'
ChestPain['index'][2] = 'Non-Anginal'
ChestPain['index'][1] = 'Attypical Anigma'
ChestPain['index'][0] = 'Typical Anigma'
ChestPain


# In[10]:


sns.set(style="darkgrid", context="talk", font_scale=0.8)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (12,3))
plt.title("The Frequency of Unique Chest Pain Types")
sns.barplot(x= ChestPain['index'], y = ChestPain['cp'], palette="pastel", alpha= 0.8)

print("As we can see, Typical Anigma is the most common type of chest pain among pateints.")


# In[11]:


ecg_data = df['restecg'].value_counts().reset_index()
ecg_data['index'][0] = 'Normal'
ecg_data['index'][1] = 'Having ST-T Wave Abnormality'
ecg_data['index'][2] = 'Showing Probable or Definite Left ventricular Hypertrophy by Estes'
ecg_data


# In[12]:


sns.set(style="darkgrid", context="talk", font_scale=0.6)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (12,3))
plt.title("ECG Data of Patients")
sns.barplot(x= ecg_data['index'], y = ecg_data['restecg'], palette="pastel", alpha= 0.8)


# In[13]:


sns.set(style="darkgrid", context="talk", font_scale=0.7)
plt.style.use("dark_background")
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5,'axes.facecolor':'black','axes.grid' : True,'grid.color': '1'})
plt.figure(figsize = (15,5))
sns.pairplot(df, hue='output', data=df, palette="pastel")


# In[14]:


plt.figure(figsize = (15,5))
plt.subplot(1, 2,1)
sns.distplot(df['trtbps'], kde=True, color = 'magenta')
plt.xlabel("Resting Blood Pressure(mmHg)")
plt.subplot(1, 2,2)
sns.distplot(df['thalachh'], kde=True, color = 'teal')
plt.xlabel("Maximum Heart Rate Achieved (bpm)")


# In[15]:


plt.subplot(1, 2,1)
sns.distplot(df['chol'], kde=True, color = 'red')
plt.xlabel("Cholestrol")


# In[16]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df)
df = scale.transform(df)
df = pd.DataFrame(df, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'caa', 'output'])
df.head()


# # We will be using the following models to predict the risk of heart attack in patients:
# 
# 
# Logistic Regression
# 
# Decision Tree
# 
# Random Forest
# 
# K Nearest Neighbour
# 
# SVM
# 

# ## Determining X and Y Values

# In[17]:


x = df.iloc[:, :-1]
x.head()


# In[18]:


y = df.iloc[:,-1:]
y.head()


# ## Splitting Data Into Training and Testing

# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=101)


# ## Logistic Regression Model

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
encoded_y= lbl.fit_transform(y_train)
logreg= LogisticRegression()
logreg.fit(x_train, encoded_y)


# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
encoded_ytest = lbl.fit_transform(y_test)
Y_pred1= logreg.predict(x_test)
Y_pred1


# In[22]:


lr_conf_matrix= confusion_matrix(encoded_ytest, Y_pred1)
lr_acc_score= accuracy_score(encoded_ytest, Y_pred1)
lr_conf_matrix


# In[23]:


print(lr_acc_score*100, "%")


# ## Decision Tree

# In[24]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, encoded_y)
ypred2= tree.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)


# In[25]:


tree_conf_matrix= confusion_matrix(encoded_ytest, ypred2)
tree_acc_score= accuracy_score(encoded_ytest, ypred2)
tree_conf_matrix


# In[26]:


print(tree_acc_score*100, "%")


# ## Random Forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, encoded_y)
ypred3= rf.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)

rf_conf_matrix= confusion_matrix(encoded_ytest, ypred3)
rf_acc_score= accuracy_score(encoded_ytest, ypred3)
rf_conf_matrix


# In[28]:


print(rf_acc_score*100, "%")


# ## K Nearest Neighbour

# In[29]:


from sklearn.neighbors import KNeighborsClassifier

error_rate= []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, encoded_y)
    pred= knn.predict(x_test)
    error_rate.append(np.mean(pred != encoded_ytest))


# In[30]:


plt.plot(range(1, 40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.xlabel("K Values")
plt.ylabel("Error Rate")
plt.title("To Find The Best Value of K")
plt.show()


# In[31]:


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train, encoded_y)
ypred4= knn.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)

knn_conf_matrix= confusion_matrix(encoded_ytest, ypred4)
knn_acc_score= accuracy_score(encoded_ytest, ypred4)
knn_conf_matrix


# In[32]:


print(knn_acc_score*100, "%")


# ## Support Vector Machine

# In[33]:


from sklearn import svm
svm = svm.SVC()

svm.fit(x_train, encoded_y)
ypred5= svm.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)

svm_conf_matrix= confusion_matrix(encoded_ytest, ypred5)
svm_acc_score= accuracy_score(encoded_ytest, ypred5)
svm_conf_matrix


# In[34]:


print(svm_acc_score*100, "%")


# ## Comparing The Accuracy of The Models

# In[35]:


model_acc = pd.DataFrame({ 'Model': ['Logistic Regression', 'Decision Tree','Random Forest', 'K Nearest Neighbor', 'SVM'], 'Accuracy': [lr_acc_score*100,tree_acc_score*100,rf_acc_score*100,knn_acc_score*100,svm_acc_score*100]})


# In[36]:


model_acc = model_acc.sort_values(by=['Accuracy'], ascending = False)


# In[37]:


model_acc


# As shown on the table above, Logistic Regression has the highest accuracy in predicting the risk of heart attack among patients

# ## AdaBoost

# In[38]:


from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(base_estimator = svm, n_estimators=100, algorithm='SAMME', learning_rate=0.01, random_state=0)


# In[39]:


adab.fit(x_train, encoded_y)
ypred6= adab.predict(x_test)
encoded_ytest = lbl.fit_transform(y_test)

adab_conf_matrix= confusion_matrix(encoded_ytest, ypred6)
adab_acc_score= accuracy_score(encoded_ytest, ypred6)
adab_conf_matrix


# In[40]:


print(adab_acc_score*100, "%")


# ## Prediction Using EvalML

# In[41]:


import evalml


# In[42]:


df = pd.read_csv("C:\\Users\\akbar\\Desktop\\Heart-Attack\\heart-data.csv")
df.head()


# In[43]:


X = df.iloc[:,:-1]
X.head()


# In[44]:


y = df.iloc[:,-1:]
y= lbl.fit_transform(y)
y


# In[78]:


X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')


# In[79]:


evalml.problem_types.ProblemTypes.all_problem_types


# In[80]:


from evalml.automl import AutoMLSearch
automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')

automl.search()


# In[48]:


automl.rankings


# In[49]:


automl.best_pipeline


# In[50]:


best_pipeline = automl.best_pipeline


# In[51]:


automl.describe_pipeline(automl.rankings.iloc[0]['id'])


# In[52]:


best_pipeline.score(X_test, y_test, objectives=["auc", "f1", "Precision","Recall"])


# In[53]:


automl_auc = AutoMLSearch(X_train=X_train, y_train=y_train,
                          problem_type='binary',
                          objective='auc',
                          additional_objectives=['f1', 'precision'],
                          max_batches=1,
                          optimize_thresholds=True)

automl_auc.search()


# In[54]:


automl_auc.rankings


# In[55]:


automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]['id'])


# In[56]:


best_pipeline_auc = automl_auc.best_pipeline


# In[57]:


best_pipeline.score(X_test, y_test, objectives=["auc"])


# In[58]:


best_pipeline_auc.save("model.pkl")


# In[59]:


final_model = automl.load('model.pkl')


# In[68]:


final_model.predict_proba(X_test)


# In[ ]:





import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('train-iris.csv')
df

x= df[['petal_length' , 'petal_width',  'sepal_length'  ,'sepal_width']]
x
y= df['class']
y

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

rf = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)
rf

ypred = rf.predict(xtest)
ypred
ytest

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypred))
conf= pd.DataFrame(confusion_matrix(ytest,ypred))
import seaborn as sns
sns.heatmap(conf,annot=True)




dt = DecisionTreeClassifier().fit(xtrain,ytrain)
dt

ypred2 = rf.predict(xtest)
ypred2
ytest

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypred2))
conf= pd.DataFrame(confusion_matrix(ytest,ypred2))
import seaborn as sns
sns.heatmap(conf,annot=True)
from sklearn import tree
tree.plot_tree(dt,fontsize=8,max_depth=3)






df2= pd.read_csv('test-iris.csv')
df2


x= df2[['petal_length' , 'petal_width',  'sepal_length'  ,'sepal_width']]
x
y= df2['class']
y

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

rf = RandomForestClassifier(n_estimators=100).fit(xtrain,ytrain)
rf

ypred = rf.predict(xtest)
ypred
ytest

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypred))
conf= pd.DataFrame(confusion_matrix(ytest,ypred))
import seaborn as sns
sns.heatmap(conf,annot=True)




dt = DecisionTreeClassifier().fit(xtrain,ytrain)
dt

ypred2 = rf.predict(xtest)
ypred2
ytest

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(ytest,ypred2))
conf= pd.DataFrame(confusion_matrix(ytest,ypred2))
import seaborn as sns
sns.heatmap(conf,annot=True)
from sklearn import tree
tree.plot_tree(dt,fontsize=8,max_depth=3)



## RF 3 


from pydataset import data
df= data('mtcars')
df
df.columns

x= df[['wt','hp']]
x
y=df['mpg']

xtrain,xtest,ytrain,ytest = train_test_split(x,y)
ytest

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100).fit(xtrain,ytrain)

ypred= rf.predict(xtest)
ypred
rf.score(xtrain,ytrain)
from sklearn.metrics import r2_score
r2_score(ytest,ypred)

from sklearn import metrics




dt = DecisionTreeRegressor().fit(xtrain,ytrain)
dt

ypred2 = dt.predict(xtest)
ypred2
ytest
r2_score(ytest,ypred2)

from sklearn.linear_model import LinearRegression
lr= LinearRegression().fit(xtrain,ytrain)
ypred3 = lr.predict(xtest)
r2_score(ytest,ypred3)



x
x.head()
new_data= [[2.1,105],[3.1,150]]
y_data = rf.predict(new_data)
y_data
y.head()
m=pd.DataFrame(new_data,index=['data1','data2'],columns=['wt','hp'])
m['mpg']=y_data
m


##   RF 4  


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
df= pd.read_csv(url,names=col_names)
df

x= df.drop('label',axis=1)
y=df['label']
x
y

from sklearn.linear_model import LogisticRegression

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)

# by RF 

rf= RandomForestClassifier(n_estimators=100)
rf = rf.fit(xtrain,ytrain)
ypred= rf.predict(xtest)
print(classification_report(ytest,ypred))

# by DT 

dt= DecisionTreeClassifier()
dt = dt.fit(xtrain,ytrain)
ypred2= dt.predict(xtest)
print(classification_report(ytest,ypred2))

# by LR 

lr=LogisticRegression().fit(xtrain,ytrain)
ypred3 = lr.predict(xtest)
ypred3
print(classification_report(ytest,ypred3))

x.head()
x_data = [[5,99,75,30,0,44,0.77,47]]
y_data= lr.predict(x_data)
y_data


ypred_prob = rf.predict_proba(xtest)[::,1]
fpr,tpr,_ = metrics.roc_curve(ytest,ypred_prob)
metrics.roc_auc_score(ytest,ypred_prob)
plt.plot(fpr,tpr)


ypred_prob = lr.predict_proba(xtest)[::,1]
fpr,tpr,_ = metrics.roc_curve(ytest,ypred_prob)
metrics.roc_auc_score(ytest,ypred_prob)
plt.plot(fpr,tpr)


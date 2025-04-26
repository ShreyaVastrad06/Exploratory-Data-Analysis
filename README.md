# Exploratory-Data-Analysis

from google.colab import files
uploaded = files.upload()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


sns.set(style="whitegrid")


df = pd.read_csv('Titanic-Dataset.csv')


print(df.head())
print(df.info())


print(df.describe())


print(df['Sex'].value_counts())
print(df['Embarked'].value_counts())



# Age distribution
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Fare distribution
plt.figure(figsize=(8,5))
df['Fare'].hist(bins=30)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Survival count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()



# Age vs Survived
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survival')
plt.show()

# Fare vs Survived
plt.figure(figsize=(8,5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare vs Survival')
plt.show()


# Correlation Matrix
plt.figure(figsize=(10,8))
numerical_df = df.select_dtypes(include=np.number)  
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.show()



fig = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age distribution by Survival")
fig.show()

fig2 = px.scatter(df, x="Age", y="Fare", color="Survived", title="Fare vs Age Scatter Plot by Survival")
fig2.show()



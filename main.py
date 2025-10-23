import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sna

data=pd.read_csv("heart.csv")

# 1. reading the data
print(data)

# 2. fetch top 4 rows 
print(f"top 5 rows  {data.head()}")

# 3. display last 5 rows
print(data.tail())

# 4. find the shape of dataset
print(f"shape of dataset {data.shape}")


# 5. get information about dataset
print(data.info())


# 6. check for missing values
print(f"missing values in dataset \n{data.isnull().sum()}")

#7. check for duplicate data and drop duplicates
duplicates = data.duplicated().any()
print(f"number of duplicate rows: {duplicates}")
data = data.drop_duplicates()
print(f"shape after removing duplicates: {data.shape}")

# 8. statistical summary of dataset
print(data.describe())

#9. draw correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sna.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#10. how many people have heart disease and how many do not have heart disease in this dataset
heart_disease_counts = data['target'].value_counts()
print(f"Heart disease counts:\n{heart_disease_counts}")
plt.figure(figsize=(6, 4))
sna.countplot(x='target', data=data)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

#11. find count of male and female in this dataset
gender_counts = data['sex'].value_counts()
print(f"Gender counts:\n{gender_counts}")
plt.figure(figsize=(6, 4))              
sna.countplot(x='sex', data=data)
plt.title("Gender Distribution")    
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

#12. find gender distribution according to the target variable.
plt.figure(figsize=(8, 6))
sna.countplot(x='sex', hue='target', data=data) 
plt.title("Gender Distribution by Heart Disease Status")       
plt.xlabel("Gender (0 = Female, 1 = Male )")
plt.ylabel("Count") 
plt.xticks([0, 1], ['Female', 'Male'])
plt.legend(title='Heart Disease', labels=['No', 'Yes']) 
plt.show()


#13. check age distribution
plt.figure(figsize=(8, 6))
sna.histplot(data['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#14. check chest pain type distribution  
plt.figure(figsize=(8, 6))
sna.countplot(x='cp', data=data)
plt.title("Chest Pain Type Distribution")   
plt.xlabel("Chest Pain Type (0-3)")
plt.ylabel("Count")
plt.xticks([0, 1, 2, 3], ['Typical angina', 'Atypical angina', 'non-anginal pain', 'asymplomatic'])
plt.show()

#15. show the chest pain distribution as per target variable
plt.figure(figsize=(8, 6))
sna.countplot(x='cp', hue='target', data=data)
plt.title("Chest Pain Type Distribution by Heart Disease Status")
plt.xlabel("Chest Pain Type (0-3)")
plt.ylabel("Count")
plt.xticks([0, 1, 2, 3], ['Typical angina', 'Atypical angina', 'non-anginal pain', 'asymplomatic'])
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

#16. show fasting blood sugar distribution according to target variable
plt.figure(figsize=(6, 4))
sna.countplot(x='fbs', hue='target', data=data)
plt.title("Fasting Blood Sugar by Heart Disease Status")
plt.xlabel("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

#17. compare resting blood pressure distribution for people with and without heart disease
plt.figure(figsize=(8, 6))
sna.boxplot(x='target', y='trestbps', data=data)
plt.title("Resting Blood Pressure by Heart Disease Status")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Resting Blood Pressure (trestbps)")
plt.show()

#18. compare cholesterol levels for people with and without heart disease
plt.figure(figsize=(8, 6))  
sna.boxplot(x='target', y='chol', data=data)
plt.title("Cholesterol Levels by Heart Disease Status")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Cholesterol (chol)")
plt.show()

# 19. compare resting blood pressure as per sex column
plt.figure(figsize=(8, 6))
sna.boxplot(x='sex',                                y='trestbps', data=data)
plt.title("Resting Blood Pressure by Gender")
plt.xlabel("Gender (0 = Female,                                         1 = Male        )")
plt.ylabel("Resting Blood Pressure (trestbps)")
plt.xticks([0, 1], ['Female', 'Male     '])
plt.show()      


#20. show distrbution of serum cholesterol
plt.figure(figsize=(8, 6))
sna.histplot(data['chol'], bins=20, kde=True)
plt.title("Serum Cholesterol Distribution")
plt.xlabel("Serum Cholesterol (chol)")
plt.ylabel("Frequency")
plt.show()



#21. plot continuos variables 
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(30, 30))
for i, var in enumerate(continuous_vars, 1):
    plt.subplot(3, 2, i)
    sna.histplot(data[var], bins=20, kde=True)
    plt.title(f"Distribution of {var}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


#22. plot categorical variables
categorical_vars = ['cp', 'restecg', 'slope', 'ca', 'thal']
plt.figure(figsize=(30, 30))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 2, i)
    sna.countplot(x=var, data=data)
    plt.title(f"Countplot of {var}")
    plt.xlabel(var)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()


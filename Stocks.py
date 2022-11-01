# Importing Relavant Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading Dataset
df =  pd.read_csv('prices.csv', header=0)
df.head()

df.drop("date", axis=1, inplace=True)

df.shape
df.describe()
df.info()

# Data Exploration
plt.figure(figsize=(16, 8))
sns.countplot(df.symbol)
plt.show()

# #### Taking Stock name and date from user to extract data from main dataset
stock_name = input("Enter Stock Name : ")
df1 = pd.DataFrame(df[(df['symbol']==stock_name)])
df1.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df1.symbol = encoder.fit_transform(df1.symbol)
df1.head()

# Spliting data into dependent and independent
x = df1.drop("volume", axis=1)
y = df1.volume

# Spliting data into train, test
from  sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Preparing Models
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

models = {
    "LinearSVR": LinearSVR(),
    "LinearRegression": LinearRegression(),
    "RandomForestRegression": RandomForestRegressor()
}

for name, model in models.items():
    # Training Model
    model.fit(x_train, y_train)
    # Displaying Accuracy
    print(name, "Accuracy", model.score(x_train, y_train))
    print("--"*50)


# Random Forest is performing much better than any other model.

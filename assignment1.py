# %%
import pandas as pd
import plotly.express as px
from prophet import Prophet

# %%
data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
data.head()

# %%
data['hourlyTime'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')
data_proph = data[['hourlyTime', 'trips']]
data_proph.columns = ['ds', 'y']

px.line(data_proph, x = 'ds', y = 'y')



# %%
model = Prophet(changepoint_prior_scale=0.5) #determines how much bend we want
modelFit = model.fit(data_proph)

# %%
futuredata = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')

futuredata['ds'] = pd.to_datetime(futuredata['Timestamp'], format='%Y-%m-%d %H:%M:%S')
futuredata_proph = futuredata[['ds']]
futuredata_proph.head()

# %%
pred = model.predict(futuredata_proph)


# %%


# %%




import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns

# Let’s add a title to test things out:
st.title('SA Municipal Budget Spent EDA')


"""
# My first app
Here's our first attempt at using data to create a table:
"""

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')


@st.cache
def load_data(fdir):
    return pd.read_csv(fdir)


df = load_data("budget-vs-actual-provincial.csv")
df

data_load_state.text('Loading data...done!')


fy_option = st.sidebar.selectbox(
    "Select Financial Year",
    list(df["financialyear"].unique()) + ["ALL"])

gov_option = st.sidebar.selectbox(
    "Select Province",
    list(df["government"].unique()) + ["ALL"])

top_department_by_spent = df.groupby(["department"]).sum().sort_values(
    by="value", ascending=False)[["value"]].head(10).index.tolist()
dp_option = st.sidebar.selectbox(
    "Select Department",
    top_department_by_spent + ["ALL"])

if gov_option == "ALL":
    data = df
else:
    data = df.loc[df.government == gov_option]

if fy_option == "ALL":
    data = data
else:
    data = data.loc[data.financialyear == fy_option]

if dp_option == "ALL":
    data = data
else:
    data = data.loc[data.department == dp_option]


data_load_state.text('Preping Viz...')

g = sns.catplot(x="financialyear", y="value",
                hue="government", kind="bar", data=data)
st.pyplot(g)

g = sns.catplot(x="financialyear", y="value", hue="department", kind="bar",
                data=data.loc[data.department.isin(top_department_by_spent)])
st.pyplot(g)

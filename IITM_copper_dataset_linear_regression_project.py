import streamlit as st
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('agg')

def load_data(session_state):
    # Read the data from Excel file
    df = pd.read_excel("D:\\Jupyter_Notebook1\\copper_set.xlsx")
    session_state.df = df
    session_state.data_loaded = True
    return df

def boxplot_selling_price(df):

    # Select the first 10,000 rows for demonstration
    df1 = df.head(10000)

    # Set the backend for plotting
    # plt.switch_backend('TkAgg')  # Or any other backend that you prefer

    # Create and display the boxplot
    fig, ax = plt.subplots()
    sns.boxplot(data=df1['selling_price'])
    st.pyplot(fig)
    # plt.show() # Displaying the chart by using matplotlib

def preprocessing(df):
    # q1 = np.quantile(df['selling_price'], 0.25)
    # q3 = np.quantile(df['selling_price'], 0.75)
    # uq = q3 + 1.5 * (q3 - q1)
    # lq = q1 - 1.5 * (q3 - q1)
    # df2 = df[(df['selling_price'] > lq) & (df['selling_price'] < uq)]

    # st.subheader("Skewness of selling_price")
    # for col in df2:
    #     st.write(col)
    #     st.write(skew(df2[col]))

    # plt.figure()
    # sns.displot(data=df2['selling_price'])
    # st.pyplot()

    # df2["selling_price"] = np.sqrt(df2["selling_price"])
    # st.subheader("Skewness of transformed selling_price")
    # st.write(skew(df2['selling_price']))

    # plt.figure()
    # sns.displot(data=df2['selling_price'])
    # st.pyplot()

    # df2.dropna(inplace=True)
    # df3 = pd.get_dummies(df2[['item type', 'status', 'material_ref']], dtype='int')
    # df3[['country', 'thickness', 'width', 'selling_price']] = df2[['country', 'thickness', 'width', 'selling_price']]

    # st.subheader("Data after preprocessing")
    # st.write(df3.head())
    # return df3


    df1 = df.head(10000)
    q1= np.quantile(df1['selling_price'],0.25)
    q3 = np.quantile(df1['selling_price'],0.75)
    uq=q3 + 1.5*(q3-q1)
    lq=q1 -1.5*(q3-q1)
    df2=df1[(df1['selling_price']>lq)&(df1['selling_price']<uq)]
    print(df2)

    for col in df2:
        print(col)

    print(skew(df2[col]))

    # fig, ax = plt.subplots()
    # sns.boxplot(data=df1['selling_price'])
    # st.pyplot(fig)

    # plt.figure()
    # fig, ax = plt.subplots()
    # sns.displot(data=df2[col])
    # st.pyplot(fig)

    df2["selling_price"]= np.sqrt(df2["selling_price"])
    skew(df2['selling_price'])

    # plt.figure()
    # sns.displot(df2['selling_price'])
    # plt.show()

    df2.sample(1000)

    df2.dropna()
    df2.isna().sum()

    df3 = pd.get_dummies(df2[['item type','status','material_ref']],dtype='int')


    df3[['country','thickness','width','selling_price']] =df2[['country','thickness','width','selling_price']]
    st.subheader("Data after preprocessing")
    st.write(df3.head())
    return df3

def train_models(df):
    X = df.drop(['selling_price'], axis=1)
    y = df['selling_price']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)
    train_pred_lr = model_lr.predict(x_train)
    test_pred_lr = model_lr.predict(x_test)

    model_lasso = Lasso()
    model_lasso.fit(x_train, y_train)
    train_pred_lasso = model_lasso.predict(x_train)
    test_pred_lasso = model_lasso.predict(x_test)

    model_ridge = Ridge()
    model_ridge.fit(x_train, y_train)
    train_pred_ridge = model_ridge.predict(x_train)
    test_pred_ridge = model_ridge.predict(x_test)

    m = (x_test, test_pred_lr, test_pred_lasso, test_pred_ridge, y_test)

    return m

def display_predictions(predictions):
    x_test, test_pred_lr, test_pred_lasso, test_pred_ridge, y_test = predictions
    x_test['pred_lr'] = test_pred_lr
    x_test['pred_lasso'] = test_pred_lasso
    x_test['pred_ridge'] = test_pred_ridge
    x_test['Original'] = y_test

    st.subheader("Predicted and Original Values")
    st.write(x_test)

# def main():
# st.title("Step-by-Step Data Analysis and Modeling")

# session_state = st.session_state
# if "data_loaded" not in session_state:
#     session_state.data_loaded = False

# # Load data button
# if st.button("Load Data"):
#     load_data(session_state)
#     st.success("Data loaded successfully!")

# # Display boxplot button
# if st.button("Display Boxplot"):
#     print(session_state)
#     if session_state.data_loaded:
#         boxplot_selling_price(session_state.df)
#         st.success("Boxplot displayed successfully!")
#     else:
#         st.error("Please load data first!")


# # Preprocessing button
# if st.button("Preprocessing"):
#     # df_processed = preprocessing(session_state.df)
#     preprocessing(session_state.df)
#     st.success("Preprocessing completed successfully!")


# # Train models button
# if st.button("Train Models"):
#     predictions = train_models(session_state.df3)
#     st.success("Models trained successfully!")

# # Display predictions button
# if st.button("Display Predictions"):
#     # display_predictions(predictions)
#     st.success("Predictions displayed successfully!")
    

# Main function
def main():
    st.title("Step-by-Step Data Analysis and Modeling")

    # Load data button
    if st.button("Load Data"):
        df = load_data(st.session_state)
        st.session_state.df = df
        st.success("Data loaded successfully!")

    # Display boxplot button
    if st.button("Display Boxplot"):
        if "df" in st.session_state:
            boxplot_selling_price(st.session_state.df)
            st.success("Boxplot displayed successfully!")
        else:
            st.error("Please load data first!")

    # Preprocessing button
    if st.button("Preprocessing"):
        if "df" in st.session_state:
            df_processed = preprocessing(st.session_state.df)
            st.session_state.df_processed = df_processed
            st.success("Preprocessing completed successfully!")
        else:
            st.error("Please load data first!")

    # Train models button
    if st.button("Train Models"):
        if "df_processed" in st.session_state:
            m = train_models(st.session_state.df_processed)
            st.session_state.m = m
            st.success("Models trained successfully!")
        else:
            st.error("Please preprocess the data first!")

    # Display predictions button
    if st.button("Display Predictions"):
        if "m" in st.session_state:
            display_predictions(st.session_state.m)
            st.success("Predictions displayed successfully!")
        else:
            st.error("Please train models first!")

if __name__ == "__main__":
    main()
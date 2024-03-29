import streamlit as st
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    df_sample=df.sample(10000)

    q1= np.quantile(df_sample['selling_price'],0.25)
    q3 = np.quantile(df_sample['selling_price'],0.75)
    uq=q3 + 1.5*(q3-q1)
    lq=q1 -1.5*(q3-q1)
    df_copper=df_sample[(df_sample['selling_price']>lq)&(df_sample['selling_price']<uq)]
    print(df_copper)

    # for col in df_copper:
    #     print(col)

    # print(skew(df_copper[col]))

    # plt.figure()
    # sns.displot(df_copper[col])
    # plt.show()

    # df_copper["selling_price"]= np.sqrt(df_copper["selling_price"])
    skew(df_copper['selling_price'])

    plt.figure()
    sns.displot(df_copper['selling_price'])
    plt.show()


    df_copper=df_copper.dropna() 
    df1=df_copper.sample(1000)
    df2= pd.get_dummies(df1[['item type','material_ref']],dtype='int')

    mapping = {'Won': 1, 'Lost': 0}
    df1['status'] = pd.to_numeric(df1['status'].map(mapping))
    df1

    df5=df1.dropna()
    df2[['status']] = df5[['status']]

    df3=df2.dropna()

    return df3


def predicting(df3):
    X=df3.drop(['status'], axis=1)
    y=df3['status']


    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    train_pred = model.predict(x_train)
    # print(train_pred)

    test_pred = model.predict(x_test)
    x_test['actual'] = y_test
    x_test['pred'] = test_pred
    # print(x_test)

    st.subheader("Predicted and Original Values")
    st.write(x_test)


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
            m = predicting(st.session_state.df_processed)
            st.session_state.m = m
            st.success("Models trained successfully!")
        else:
            st.error("Please preprocess the data first!")


if __name__ == "__main__":
    main()
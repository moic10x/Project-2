import streamlit as st
import numpy as np
import pandas as pd
st.write("# Machine Learning Project Findings")
st.write("Hi, this is our presentation using Streamlit. Unfortunately, we couldn't connect to our main GC notebook so we had to start from scratch.")

st.write("The main goal of our project was to see if we could use historical data to predict housing prices in eight selected MSAs.")






st.write("We decided to use MAPE instead of MSE simply because our targets were large quantities, and it was helpful to think in terms of percentages.")

st.write("Here is an example of some visual feedback we were getting, the loss function for our model run on Chicago")
st.image('/Users/willcolwell/Desktop/Screenshot 2023-07-06 at 10.09.44 AM.png')

st.write("Histograms were also useful to look, to see where error was falling.")


st.write("This was the histogram of errors for Chicago. Without this view, we never would have seen the outlier guesses.")
st.image('/Users/willcolwell/Desktop/Screenshot 2023-07-06 at 10.11.53 AM.png')

st.write("However, as you can tell from the MAPE this neural net was pretty useless.")
st.write("A 20% plus error in terms of predicting housing price is completely unactionable...")


st.write("Instead of throwing a complex model at our relatively simple data, we decided to go back to square one and use a simple model!")

st.write("Below you can see a K Nearest Neighbors model compared to two different neural nets in terms of MAPE:")


st.write('MAPEs for different models:')
df = pd.DataFrame({'City': ['Miami', 'Chicago', 'Seattle', 'Houston', 'New York', 'Phoenix', 'Boston', 'Los Angeles'],
                   'One_Layer_MAPE': [29.4418, 39.5569, 34.4247, 43.7381, 13.5537, 45.5187, 27.3373, 50.2665],
                    'Two_Layer_MAPE': [21.0581, 18.3848, 31.9227, 31.0005, 23.4340, 63.7088, 36.8664, 36.9379],
                    'KNN Model_MAPE': [25.3777, 07.2884, 06.8286, 24.5752, 06.8825, 29.3619, 18.9692, 18.9797]})
st.write(df)

st.write("Wow! Looks significantly better. A strong lesson in not putting a Rolls Royce Engine in a Honda Prius")

st.write("but can we do better......")


st.header("Hyperparamter tuning!")

st.write("A popular way to optimize hyperparameters with KNN is something called GridSearchCV (essentially a cross-validation technique.)")

st.write("Using GridSearchCV from sklearn we found the below:")

st.image('/Users/willcolwell/Desktop/Screenshot 2023-07-06 at 5.22.11 PM.png')


st.write("Let's look at the results with a new leaf size and number of neighbors")

df = pd.DataFrame({'City': ['Miami', 'Chicago', 'Seattle', 'Houston', 'New York', 'Phoenix', 'Boston', 'Los Angeles'],
                   'One_Layer_MAPE': [29.4418, 39.5569, 34.4247, 43.7381, 13.5537, 45.5187, 27.3373, 50.2665],
                    'Two_Layer_MAPE': [21.0581, 18.3848, 31.9227, 31.0005, 23.4340, 63.7088, 36.8664, 36.9379],
                    'KNN Model_MAPE': [22.1728, 08.5710, 22.068, 22.8320, 06.325, 30.5834, 16.4761, 21.43],
                    'KNN_Model_Optimized_MAPE': [23.732, 09.737, 22.3877, 23.8622, 06.11, 31.5044, 17.2854, 18.9200]})
st.write(df)


st.write("Which model won?")

st.code('knn_initial = [22.1728, 08.5710, 22.068, 22.8320, 06.325, 30.5834, 16.4761, 21.43] knn_optimal = [23.732, 09.737, 22.3877, 23.8622, 06.11, 31.5044, 17.2854, 18.9200] new_list = [a_i - b_i for a_i, b_i in zip(knn_initial, knn_optimal)]')

st.image('/Users/willcolwell/Desktop/Screenshot 2023-07-06 at 5.36.38 PM.png')

st.write('The default parameters win... looks like we just need better data.')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

st.title("MBA AND RS")

uploaded_file = st.file_uploader("sales_data.csv", type=["csv"])

if uploaded_file is not None:
    sales_data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(sales_data.head())
    
    
    if 'InvoiceDate' in sales_data.columns:
        sales_data['InvoiceDate'] = pd.to_datetime(sales_data['InvoiceDate'])

    sales_data.dropna(inplace=True)
    
    
    st.subheader("Market Basket Analysis")
    if 'InvoiceNo' in sales_data.columns and 'StockCode' in sales_data.columns:
        basket = sales_data.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)
        
        frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
        
        st.write("Frequent Itemsets:")
        st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False))
        
        st.write("Association Rules:")
        st.dataframe(rules.sort_values(by='lift', ascending=False))
        
        # Visualization
        st.subheader("Support vs. Confidence Scatter Plot")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=rules['support'], y=rules['confidence'], hue=rules['lift'], size=rules['lift'], palette='viridis')
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Support vs. Confidence")
        st.pyplot(plt)
    else:
        st.error("Required columns 'InvoiceNo' and 'StockCode' are missing in the dataset.")
    
    
    st.subheader("Recommendation System using SVD")
    if {'CustomerID', 'StockCode', 'Quantity'}.issubset(sales_data.columns):
        reader = Reader(rating_scale=(1, sales_data['Quantity'].max()))
        data = Dataset.load_from_df(sales_data[['CustomerID', 'StockCode', 'Quantity']], reader)
        trainset = data.build_full_trainset()
        
        model = SVD()
        cross_validate(model, data, cv=5)
        model.fit(trainset)
        
        customer_id = st.number_input("Enter Customer ID for recommendations:", min_value=int(sales_data['CustomerID'].min()), max_value=int(sales_data['CustomerID'].max()))
        
        if st.button("Get Recommendations"):
            product_ids = sales_data['StockCode'].unique()
            predictions = [model.predict(customer_id, pid).est for pid in product_ids]
            recommendations = pd.DataFrame({'StockCode': product_ids, 'Predicted Rating': predictions})
            recommendations = recommendations.sort_values(by='Predicted Rating', ascending=False).head(10)
            
            st.write("Top 10 Recommended Products:")
            st.dataframe(recommendations)
    else:
        st.error("Required columns 'CustomerID', 'StockCode', and 'Quantity' are missing in the dataset.")

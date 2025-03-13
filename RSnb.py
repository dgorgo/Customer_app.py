{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "94b19cc7-e648-4aba-a945-735da8498005",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d50c2378-f0c5-496f-8091-328dc76f511d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import networkx as nx\n",
    "import plotly.graph_objects as go"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5eae020a-3d35-4258-baf5-2d3817d8a367",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7e198bbf-53ce-433d-9521-6967b16ae308",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from mlxtend.frequent_patterns import apriori, association_rules\n",
    "from surprise import Dataset, Reader, SVD\n",
    "from surprise.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f86282ee-4314-4733-bfc5-5b981e7a656c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-03-13 16:52:55.381 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\etietop.udofia\\AppData\\Local\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Streamlit UI\n",
    "st.title(\"Magic Store\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "39f912e9-7a06-4f50-9f1a-db5c7ca1cf41",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\etietop.udofia\\AppData\\Local\\Temp\\ipykernel_27456\\3357326288.py:1: DtypeWarning: Columns (5) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  sales_data = pd.read_csv('sales_data.csv')\n"
     ]
    }
   ],
   "source": [
    "sales_data = pd.read_csv('sales_data.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d5ac16d5-904c-482a-acc0-5a85c04a66bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "uploaded_file = st.file_uploader(\"Upload Sales Data (CSV)\", type=[\"csv\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8118dd68-fe68-47b6-b31e-12ed69a178dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "if uploaded_file:\n",
    "    sales_data = pd.read_csv('sales_data.csv')\n",
    "    st.write(\"Data Preview:\", sales_data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "66cc88da-6229-4d71-96cb-5c4435f26e2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "sales_data['Quantity'] = sales_data['Delivered Qty']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "53cc5eed-ef91-45b5-9759-bf4882fb4796",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\etietop.udofia\\AppData\\Local\\Temp\\ipykernel_27456\\1220929609.py:3: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.\n",
      "  basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert to 1/0\n"
     ]
    }
   ],
   "source": [
    "# Data Preprocessing for MBA\n",
    "basket = sales_data.pivot_table(index='Customer_Name', columns='SKU_Code', values='Quantity', aggfunc='sum').fillna(0)\n",
    "basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Convert to 1/0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "72f693f8-ad38-43cb-8cc0-7d70100f5a5f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\etietop.udofia\\AppData\\Local\\anaconda3\\Lib\\site-packages\\mlxtend\\frequent_patterns\\fpcommon.py:161: DeprecationWarning: DataFrames with non-bool types result in worse computationalperformance and their support might be discontinued in the future.Please use a DataFrame with bool type\n",
      "  warnings.warn(\n",
      "2025-03-13 17:00:22.117 Serialization of dataframe to Arrow table was unsuccessful due to: (\"Could not convert frozenset({'10000001'}) with type frozenset: did not recognize Python value type when inferring an Arrow data type\", 'Conversion failed for column antecedents with type object'). Applying automatic fixes for column types to make the dataframe Arrow-compatible.\n"
     ]
    }
   ],
   "source": [
    "# Market Basket Analysis using Apriori\n",
    "frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)\n",
    "rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)\n",
    "st.subheader(\"Association Rules\")\n",
    "st.write(rules.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "b32f71e6-3925-4dd1-a1f5-8cb71e7cddf1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Preprocessing for Recommendation System\n",
    "reader = Reader(rating_scale=(0, 5))  # Adjust scale if needed\n",
    "data = Dataset.load_from_df(sales_data[['Customer_Name', 'SKU_Code', 'Quantity']], reader)\n",
    "trainset, testset = train_test_split(data, test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "7dba6d89-dafb-4efa-98a1-e180293c490a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<surprise.prediction_algorithms.matrix_factorization.SVD at 0x25a19baae70>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Collaborative Filtering using SVD\n",
    "model = SVD()\n",
    "model.fit(trainset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "29e2c727-3b9b-4a3d-96e1-3df018cb26ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Recommendations\n",
    "def recommend_products(customer_id, top_n=5):\n",
    "    all_skus = sales_data['SKU_Code'].unique()\n",
    "    predictions = [model.predict(customer_id, sku) for sku in all_skus]\n",
    "    top_skus = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]\n",
    "    return [pred.iid for pred in top_skus]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "304d85ff-b169-4d62-b279-3bd91053e4f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "customer = st.selectbox(\"Select a Customer\", sales_data['Customer_Name'].unique())\n",
    "if st.button(\"Get Recommendations\"):\n",
    "    recommendations = recommend_products(customer)\n",
    "    st.subheader(f\"Recommended Products for {customer}\")\n",
    "    st.write(recommendations)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "39688c15-517d-42b2-bf6f-60ef1ebb4aea",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}



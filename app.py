from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('kmeans_model.pkl', 'rb'))

def load_and_clean_data(file_path):
    # Load data
    retail = pd.read_csv('OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)
    
    # Convert customerid to string and create amount column
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Compute RFM matrices
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
    rfm_f = rfm_f.reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M')
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Removing (statistical) outliers for Amount
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5 * IQR) & (rfm.Amount <= Q3 + 1.5 * IQR)]
    
    return rfm


def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    # Instantiate StandardScaler
    scaler = StandardScaler()

    # Fit and transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled, columns=['Amount', 'Frequency', 'Recency'])

    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)
    df = preprocess_data(file_path)[1]
    results_df = model.predict(df)
    df_with_id = preprocess_data(file_path)[0]

    df_with_id['clusters_id'] = results_df

    # Generate and save images
    sns.stripplot(x='clusters_id', y ='Amount', data=df_with_id, hue='clusters_id')
    amount_img_path ='static/clustersId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()
    
    sns.stripplot(x='clusters_id', y ='Frequency', data=df_with_id, hue='clusters_id')
    freq_img_path ='static/clusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()
    
    sns.stripplot(x='clusters_id', y ='Recency', data=df_with_id, hue='clusters_id')
    recency_img_path ='static/clusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    # Render result.html with image paths
    return render_template('result.html', 
                           amount_image=amount_img_path,
                           freq_img=freq_img_path,
                           recency_img=recency_img_path)

if __name__ == "__main__":
    app.run(debug=True)

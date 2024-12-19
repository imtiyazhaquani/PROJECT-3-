#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_excel('World_development_mesurement.xlsx')
df


# In[3]:


# EDA (BASIC EXPLORATION)
df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum().sum()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.describe().T


# In[9]:


df = df.drop(['Ease of Business', 'Number of Records'], axis=1)


# In[10]:


df


# In[11]:


# CHANGING THE DTYPES

object_columns = df.select_dtypes(include=['object']).columns
print(object_columns)


# In[12]:


for col in object_columns:
    df[col] = df[col].str.replace(r'[^\d.-]', '', regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')


# In[13]:


df


# In[14]:


df.info()


# In[15]:


# DETECT OUTLIERS USING IQR

outliers_summary = {}

def detect_outliers_iqr(data, column):
  Q1 = data[column].quantile(0.25)
  Q3 = data[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
  outliers_summary[column] = len(outliers)
  return outliers_summary

for column, count in outliers_summary.items():
  print(f"Number of outliers in {column}: {count}")


# In[16]:


# VISUALIZATIONS

# 1. CORRELATION MATRIX
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
plt.title('Global Development Measurement Data Correlation Matrix')
plt.show()


# In[17]:


# HISTOGRAMS

plt.figure(figsize=(20, 15))

# List of numerical columns in the DataFrame
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Number of columns and rows to arrange the subplots
num_columns = 3
num_rows = (len(numerical_columns) // num_columns) + 1

# Create subplots
for i, col in enumerate(numerical_columns, start=1):
    plt.subplot(num_rows, num_columns, i)
    sns.histplot(data=df, x=col, kde=True, bins=30, color='blue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[18]:


# SCATTER PLOT MATRIX (PAIRPLOT)

# List of numerical columns to include in the pairplot.
numerical_cols = ['Birth Rate', 'Business Tax Rate', 'CO2 Emissions', 'Days to Start Business', 'Energy Usage', 'GDP', 'Health Exp % GDP', 'Health Exp/Capita', 'Hours to do Tax', 'Infant Mortality Rate', 'Internet Usage', 'Lending Interest', 'Life Expectancy Female', 'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14', 'Population 15-64', 'Population 65+', 'Population Total', 'Population Urban', 'Tourism Inbound', 'Tourism Outbound']

sns.pairplot(df[numerical_cols])
plt.suptitle('Scatter Plot Matrix of Numerical Features', y=1.02) # Add a title to the entire plot
plt.show()


# In[19]:


# COMPARISION CHART (SCATTER PLOT)

# 'Birth_Rate' vs 'Infant_Mortality_Rate'
sns.pairplot(df, x_vars=['Birth Rate'], y_vars=['Infant Mortality Rate'],height=5)

plt.xlabel('Birth Rate')
plt.ylabel('Infant Mortality Rate')
plt.title('Scatterplot of Birth Rate vs Infant Mortality Rate')
plt.show()

# CO2 Emissions vs Energy
sns.scatterplot(x='CO2 Emissions', y='Energy Usage',data=df)

plt.xlabel('CO2 Emissions')
plt.ylabel('Energy Usage')
plt.title('Scatterplot of CO2 Emissions vs Energy Usage ')
plt.show()

# Total Population vs GDP
sns.scatterplot(x='Population Total', y= 'GDP', data=df)
plt.title('Total Population vs GDP')
plt.show()

# Tourism Inbound vs Tourism Outbound
sns.scatterplot(x='Tourism Inbound', y='Tourism Outbound', data=df)
plt.title('Tourism Inbound vs Tourism Outbound')
plt.show()


# In[20]:


# BOXPLOTS

# Set the size of the entire figure
plt.figure(figsize=(18, 12))

# List of columns to plot
columns_to_plot = ['Birth Rate', 'Business Tax Rate', 'CO2 Emissions', 'Days to Start Business', 'Energy Usage', 'GDP', 'Health Exp % GDP', 'Health Exp/Capita', 'Hours to do Tax', 'Infant Mortality Rate', 'Internet Usage', 'Lending Interest', 'Life Expectancy Female', 'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14', 'Population 15-64', 'Population 65+', 'Population Total', 'Population Urban', 'Tourism Inbound', 'Tourism Outbound']

# Calculate the number of rows and columns needed for subplots
num_cols = 3
num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

# Create a subplot for each column
for i, col in enumerate(columns_to_plot, start=1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[21]:


# REMOVING OUTLIERS

for cols in df.columns:
    if cols != 'Country':
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[cols] = np.where(df[cols] < lower_bound, lower_bound, df[cols])
        df[cols] = np.where(df[cols] > upper_bound, upper_bound, df[cols])

        print(f"Outliers removed from {cols}")


# In[22]:


df_box=df.drop('Country', axis=1)
plt.figure(figsize=(18,40))
for i, j in enumerate(df_box.columns):
    plt.subplot(25,1,i+1)
    sns.boxplot(x=df_box[j])
    plt.xlabel(j)
    plt.tight_layout()


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# In[24]:


df


# In[25]:


from sklearn.impute import SimpleImputer

# Initialize the imputer to replace NaNs with the mean of the column
imputer = SimpleImputer(strategy='mean')

# Apply imputer to the dataset
scaled_data_imputed = imputer.fit_transform(df)


# In[26]:


pc = PCA()
pc_components = pc.fit_transform(scaled_data_imputed)


# In[27]:


# in percentage - The amount of variance that each PCA explains is
var = pc.explained_variance_ratio_
var


# In[28]:


# Cumulative variance
var1 = np.cumsum(np.round(var,decimals=4)*100)
var1


# In[29]:


df_pca = pc_components[:,:15]


# In[30]:


## Plot between PCA's
x=pc_components[:,0]
y=pc_components[:,1]
z=pc_components[:,2]
plt.scatter(x,y)
plt.scatter(x,z)
plt.scatter(y,z)
plt.show()


# In[31]:


#Kmeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[32]:


## creating clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_pca)


# In[33]:


plt.scatter(df_pca[y_kmeans == 0, 0], df_pca[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df_pca[y_kmeans == 1, 0], df_pca[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df_pca[y_kmeans == 2, 0], df_pca[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of measurements')
plt.legend()
plt.show()


# In[34]:


## Accuracy check

s1_kmeans = silhouette_score(df_pca, y_kmeans)
print('Silhouette Score for K-means clustring :', s1_kmeans)


# In[35]:


#hierarchical clustering
dendrogram = sch.dendrogram(sch.linkage(df_pca, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[36]:


hc = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
y_hc = hc.fit_predict(df_pca)


# In[37]:


plt.scatter(df_pca[y_hc == 0, 0], df_pca[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df_pca[y_hc == 1, 0], df_pca[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df_pca[y_hc == 2, 0], df_pca[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of measurments')
plt.legend()
plt.show()


# In[38]:


## Accuracy check
s1_hierarchy = silhouette_score(df_pca,y_hc)
print('Silhouette Score for Hierarchy clustring :',s1_hierarchy)


# In[39]:


#DBSCAN
eps = 0.3
min_samples = 3
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Fit the data to obtain clustering labels
dbscan_labels = dbscan.fit_predict(df_pca)


# In[40]:


plt.scatter(df_pca[:, 0], df_pca[:, 1], c=dbscan_labels)
plt.show()


# In[41]:


df['cluster']=dbscan.labels_
df.head()


# In[42]:


# Use pandas filtering and get noisy datapoints -1
df[df['cluster']==-1]


# In[43]:


s1_dbscan = silhouette_score(df_pca, dbscan_labels)
print("Silhouette Score for DBSCAN is:", s1_dbscan)


# In[44]:


# Initialize t-SNE with default parameters
tsne = TSNE()

# Fit and transform the data to 2 dimensions
df_tsne = tsne.fit_transform(scaled_data_imputed)

# Plot the results
plt.scatter(df_tsne[:, 0], df_tsne[:, 1])
plt.show()


# In[45]:


#Kmeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_tsne)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[46]:


# Perform clustering with KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_tsne)


# In[47]:


# Plot the clusters
plt.scatter(df_tsne[y_kmeans == 0, 0], df_tsne[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df_tsne[y_kmeans == 1, 0], df_tsne[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df_tsne[y_kmeans == 2, 0], df_tsne[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of measurments')
plt.legend()
plt.show()


# In[48]:


## Accuracy check
s3_kmeans = silhouette_score(df_tsne, y_kmeans)
print('Silhouette Score for K-means clustring :', s3_kmeans)


# In[49]:


#hierarchi
dendrogram = sch.dendrogram(sch.linkage(df_tsne, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[50]:


plt.scatter(df_tsne[y_hc == 0, 0], df_tsne[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(df_tsne[y_hc == 1, 0], df_tsne[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(df_tsne[y_hc == 2, 0], df_tsne[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(df_tsne[y_hc == 3, 0], df_tsne[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.title('Clusters of measurments')
plt.legend()
plt.show()


# In[51]:


## Accuracy check
s3_hierarchy = silhouette_score(df_tsne,y_hc)
print('Silhouette Score for Hierarchy clustring :',s3_hierarchy)


# In[52]:


# Evaluation
df_ = pd.DataFrame({'Method':['pca_kmeans','pca_hierarchy','pca_DBSCAN','tsne_kmeans','tsne_hierarchy'],
                   'Silhouette Score':[s1_kmeans,s1_hierarchy,s1_dbscan,s3_kmeans,s3_hierarchy]})
df_


# In[53]:


data = {
    "Method": ["pca_kmeans", "pca_hierarchy", "pca_DBSCAN", "tsne_kmeans", "tsne_hierarchy"],
    "Silhouette Score": [0.725720, 0.720304, 0.394177, 0.385464, 0.398848]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Sort the data for better visualization
df = df.sort_values(by="Silhouette Score", ascending=False)

# Plot
plt.figure(figsize=(8, 4))
plt.barh(df["Method"], df["Silhouette Score"], color="skyblue")
plt.xlabel("Silhouette Score")
plt.ylabel("Clustering Methods")
plt.title("Silhouette Scores for Different Clustering Methods")
plt.gca().invert_yaxis()  # Highest score at the top
plt.show()


# In[54]:


#final model
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(df_pca)


# In[55]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
from flask import Flask, request, jsonify


# In[56]:


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(df_pca)  


# In[57]:


# Save the model to a file
joblib.dump(kmeans, 'kmeans_model.pkl')


# In[58]:


# Load the model for deployment
model = joblib.load('kmeans_model.pkl')


# In[ ]:


# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask Machine Learning API! Use the '/predict' endpoint to get cluster predictions."

@app.route('/predict', methods=['POST'])
def predict_cluster():
    try:
        # Parse input data
        input_data = request.json
        if not input_data or 'features' not in input_data:
            return jsonify({'error': 'Invalid input. Provide "features" as a list.'}), 400

        features = np.array(input_data['features']).reshape(1, -1)

        # Predict cluster
        cluster = model.predict(features)[0]

        return jsonify({'cluster': int(cluster)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))  # Default to port 5001, but allow override
    try:
        app.run(debug=False, host='0.0.0.0', port=port)
    except OSError as e:
        print(f"Error: {e}. Port {port} may already be in use.")


# In[ ]:





# In[ ]:





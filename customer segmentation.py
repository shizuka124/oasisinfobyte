import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('/Users/arya/PycharmProjects/pythonProject/oasis/ifood_df.csv')

# Display initial information about the data
print(data.head())
print(data.columns)
print(data.isna().sum())
print(data.info())
print(data.nunique())

# Drop unnecessary columns
data.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

# Plot a boxplot for MntTotal
plt.figure(figsize=(6, 4))
sns.boxplot(data=data, y='MntTotal')
plt.title('Box Plot for MntTotal')
plt.ylabel('MntTotal')
plt.show()

# Identify and remove outliers
Q1 = data['MntTotal'].quantile(0.25)
Q3 = data['MntTotal'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['MntTotal'] > lower_bound) & (data['MntTotal'] < upper_bound)]
print(data.describe())

# Plot boxplot and histogram for Income
plt.figure(figsize=(6, 4))
sns.boxplot(data=data, y='Income', palette='viridis')
plt.title('Box Plot for Income')
plt.ylabel('Income')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Income', bins=30, kde=True)
plt.title('Histogram for Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Plot histogram for Age
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Age', bins=30, kde=True)
plt.title('Histogram for Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Calculate skewness and kurtosis for Age
print("Skewness: %f" % data['Age'].skew())
print("Kurtosis: %f" % data['Age'].kurt())

# Define column groups
cols_demographics = ['Income', 'Age']
cols_children = ['Kidhome', 'Teenhome']
cols_marital = ['marital_Divorced', 'marital_Married', 'marital_Single', 'marital_Together', 'marital_Widow']
cols_mnt = ['MntTotal', 'MntRegularProds', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
cols_communication = ['Complain', 'Response', 'Customer_Days']
cols_campaigns = ['AcceptedCmpOverall', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
cols_source_of_purchase = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
cols_education = ['education_2n Cycle', 'education_Basic', 'education_Graduation', 'education_Master', 'education_PhD']

# Calculate and plot correlation matrix heatmap
corr_matrix = data[['MntTotal'] + cols_demographics + cols_children].corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Calculate point-biserial correlation for marital status and education with MntTotal
for col in cols_marital:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation:.4f}: Point-Biserial Correlation for {col} with p-value {p_value:.4f}')

for col in cols_education:
    correlation, p_value = pointbiserialr(data[col], data['MntTotal'])
    print(f'{correlation:.4f}: Point-Biserial Correlation for {col} with p-value {p_value:.4f}')

# Function to determine marital status
def get_marital_status(row):
    if row['marital_Divorced'] == 1:
        return 'Divorced'
    elif row['marital_Married'] == 1:
        return 'Married'
    elif row['marital_Single'] == 1:
        return 'Single'
    elif row['marital_Together'] == 1:
        return 'Together'
    elif row['marital_Widow'] == 1:
        return 'Widow'
    else:
        return 'Unknown'

data['Marital'] = data.apply(get_marital_status, axis=1)

# Plot MntTotal by marital status
plt.figure(figsize=(8, 6))
sns.barplot(x='Marital', y='MntTotal', data=data, palette='viridis')
plt.title('MntTotal by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('MntTotal')
plt.show()

# Function to determine relationship status
def get_relationship(row):
    if row['marital_Married'] == 1 or row['marital_Together'] == 1:
        return 1
    else:
        return 0

data['In_relationship'] = data.apply(get_relationship, axis=1)

# Standardize data and perform PCA for clustering
scaler = StandardScaler()
cols_for_clustering = ['Income', 'MntTotal', 'In_relationship']
data_scaled = data.copy()
data_scaled[cols_for_clustering] = scaler.fit_transform(data[cols_for_clustering])
print(data_scaled[cols_for_clustering].describe())

pca = PCA(n_components=2)
pca_res = pca.fit_transform(data_scaled[cols_for_clustering])
data_scaled['pc1'] = pca_res[:, 0]
data_scaled['pc2'] = pca_res[:, 1]
X = data_scaled[cols_for_clustering]

# Determine optimal number of clusters using inertia and silhouette score
inertia_list = []
silhouette_list = []

for K in range(2, 10):
    model = KMeans(n_clusters=K, random_state=7)
    clusters = model.fit_predict(X)
    inertia_list.append(model.inertia_)
    silhouette_list.append(silhouette_score(X, clusters))

# Plot inertia and silhouette score
plt.figure(figsize=(7, 5))
plt.plot(range(2, 10), inertia_list, color=(54/255, 113/255, 130/255))
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(2, 10), silhouette_list, color=(54/255, 113/255, 130/255))
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# Perform clustering with optimal number of clusters
optimal_k = 4
model = KMeans(n_clusters=optimal_k, random_state=7)
data_scaled['Cluster'] = model.fit_predict(data_scaled[cols_for_clustering])
data['Cluster'] = data_scaled['Cluster']

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pc1', y='pc2', data=data_scaled, hue='Cluster', palette='viridis')
plt.title('Clustered Data Visualization')
plt.xlabel('Principal Component 1 (pc1)')
plt.ylabel('Principal Component 2 (pc2)')
plt.legend(title='Clusters')
plt.show()

# Group data by clusters and calculate mean values for each cluster
cluster_summary = data.groupby('Cluster')[cols_for_clustering + cols_mnt].mean().reset_index()

# Melt the data for better visualization
melted_data = pd.melt(cluster_summary, id_vars="Cluster", var_name="Product", value_name="Consumption")

# Plot product consumption by cluster
plt.figure(figsize=(12, 6))
sns.barplot(x="Cluster", y="Consumption", hue="Product", data=melted_data, ci=None, palette="viridis")
plt.title("Product Consumption by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Product Consumption")
plt.xticks(rotation=0)
plt.legend(title="Product", loc="upper right")
plt.show()

# Plot cluster sizes
cluster_sizes = data.groupby('Cluster')[['MntTotal']].count().reset_index()
cluster_sizes.columns = ['Cluster', 'Count']
cluster_sizes['Share%'] = (cluster_sizes['Count'] / len(data) * 100).round(0)

plt.figure(figsize=(8, 6))
sns.barplot(x='Cluster', y='Count', data=cluster_sizes, palette='viridis')
plt.title('Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Plot income distribution by cluster
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Income', data=data, palette='viridis')
plt.title('Income by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Income')
plt.show()

# Plot MntTotal vs Income by cluster
plt









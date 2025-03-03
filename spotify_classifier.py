# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load Dataset
file_path = "C:/Users/ADMIN/Documents/datasets/spotify2.csv"  
df = pd.read_csv(file_path)

df.head()
df.tail()


#check the number of rows and columns
df.shape

#EDA and Data Cleaning
df.info()

# Drop unnecessary columns
df.drop(columns=['index', 'track_id', 'artists', 'album_name', 'track_name'], inplace=True)

#check if all columns in the dataset have consistent data types
df.dtypes.value_counts()

#Missing Values
df.isnull().sum()

#check the number of duplicate rows in the dataset
print(f"{len(df[df.duplicated()])} entries are duplicates")

#removing duplicates
df = df.drop_duplicates().reset_index(drop=True)

#recheck the number of duplicate rows in the dataset
print(f"{len(df[df.duplicated()])} entries are duplicates")

df.nunique()

# Check the range of time_signature
df['time_signature'].value_counts()

# Drop values with '0' as a time signature of 0 is not possible in music
df = df[df['time_signature'] != 0]

# Remove tracks with duration = 0
df = df[df['duration_ms'] > 0]

# Convert duration from ms to minutes
df['duration_min'] = df['duration_ms'] / 60000  
df = df.drop(columns=['duration_ms'])

# Display updated dataset information
df.info()
print(f"Final dataset size: {df.shape}")
print(df.head())

#Create the Mood Column
def classify_mood(row):
    if row['valence'] > 0.6 and row['energy'] > 0.6:
        return 'Happy'
    elif row['valence'] < 0.4 and row['energy'] < 0.4:
        return 'Sad'
    elif row['energy'] > 0.7 and row['danceability'] > 0.6:
        return 'Energetic'
    elif row['energy'] < 0.4 and row['acousticness'] > 0.6:
        return 'Calm'
    else:
        return 'Chill'  # Default for songs that don't fit other categories

df['mood'] = df.apply(classify_mood, axis=1)
df['mood'].nunique()

df.info()

#Visualize Mood Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df['mood'], palette='viridis')
plt.title('Mood Distribution of Songs')
plt.xlabel('Mood')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Encode the 'mood' column
le_mood = LabelEncoder()
df['mood_encoded'] = le_mood.fit_transform(df['mood'])

# Encode the 'track_genre' column
le_genre = LabelEncoder()
df['track_genre_encoded'] = le_genre.fit_transform(df['track_genre'])

# Convert 'explicit' (Boolean) into integer (0 or 1)
df['explicit'] = df['explicit'].astype(int)

df = df.drop(columns=['mood', 'track_genre'])

df.info()

# Compute correlation matrix
corr_matrix = df.corr()

# Print correlation values with the target variable (mood_encoded)
print("Correlation with mood_encoded:\n", corr_matrix["mood_encoded"].sort_values(ascending=False))

# Plot full correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

#outlier detection using boxplots
plt.figure(figsize=(12, 6))
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_numeric.boxplot(rot=45, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Numeric Features")
plt.show()

#outlier detection using z_score
z_scores = np.abs(zscore(df_numeric))
threshold = 3
outliers = (z_scores > threshold).sum()
print(f"Number of outliers per feature:\n{outliers}")


#outlier detection using Scatter Plots for Key Features
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['energy'], y=df['valence'], hue=df['mood_encoded'], palette='viridis')
plt.title("Energy vs Valence with Mood Encoding")
plt.show()

#filter out outliers
threshold = 5  # Adjusted threshold
z_scores = np.abs(zscore(df_numeric))
df_filtered = df[(z_scores < threshold).all(axis=1)]

print(f"Original dataset size: {df.shape}")
print(f"Filtered dataset size: {df_filtered.shape}")

df_final = df_filtered.copy()

#Apply winsorization/capping
# Cap the extreme 1% of values
df_final['loudness'] = winsorize(df_final['loudness'], limits=[0.01, 0.01])
df_final['speechiness'] = winsorize(df_final['speechiness'], limits=[0.01, 0.01])
df_final['duration_min'] = winsorize(df_final['duration_min'], limits=[0.01, 0.01])

# Recheck Z-scores after capping
z_scores_capped = np.abs(zscore(df_final.select_dtypes(include=['float64', 'int64'])))
outliers_capped = (z_scores_capped > 5).sum()

print(f"Number of outliers per feature after capping:\n{outliers_capped}")

#Recheck outliers after capping
plt.figure(figsize=(12, 6))
df_final.boxplot(rot=45, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("Boxplot of Numeric Features After Winsorization")
plt.show()

# Separate features and target before scaling
X = df_final.drop(columns=['mood_encoded'])  # Features
y = df_final['mood_encoded'].astype(int)  # Ensure target remains categorical (integer)

# Feature Scaling (Apply StandardScaler only to X)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Recalculate Z-scores after scaling
z_scores_scaled = np.abs(zscore(X_scaled))  # Compute Z-scores for scaled features
outliers_scaled = (z_scores_scaled > 5).sum()  # Count outliers
print(f"Number of outliers per feature after scaling:\n{outliers_scaled}")


#Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,  # Use X_scaled instead of X
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensures balanced class distribution
)

# Train RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get Feature Importance
feature_importances = pd.DataFrame({"Feature": X_train.columns, "Importance": rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importances, palette="viridis")
plt.title("Feature Importance - RandomForest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# Selecting only the top features based on RandomForest importance
top_features = ['valence', 'energy', 'danceability', 'acousticness', 'loudness', 'mood_encoded']
df_selected = df_final[top_features]

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected.drop(columns=['mood_encoded'])), 
                         columns=['valence', 'energy', 'danceability', 'acousticness', 'loudness'])

# Target variable (Ensure it's categorical)
X = df_scaled
y = df_selected['mood_encoded'].astype(int)  # Converting to integer categories

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning for XGBoost
xgb_model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='mlogloss', 
    random_state=42,
    n_estimators=10,  # less trees with lower learning rate
    learning_rate=0.03,  # Lower learning rate
    max_depth=2,  # Control tree complexity
    lambda_=100,  # L2 regularization
    alpha=3  # L1 regularization
)

# Hyperparameter tuning for LGBM
lgbm_mo0del = LGBMClassifier(
    random_state=42,
    n_estimators=10,
    learning_rate=0.04,
    max_depth=5,
    num_leaves=10,  # Avoid too many leaves
    reg_alpha=5,  # L1 regularization
    reg_lambda=5,  # L2 regularization
    feature_fraction=0.8,  # Use only 80% of features per iteration
    bagging_fraction=0.8,  # Use only 80% of samples per iteration
    bagging_freq=5  # Perform bagging every 5 iterations
)

# Train and evaluate XGBoost
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start_time

y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

# Train and evaluate LGBM
start_time = time.time()
lgbm_model.fit(X_train, y_train)
lgbm_time = time.time() - start_time

y_pred_lgbm = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)

# Print results
results = {
    "Model": ["XGBoost", "LGBM"],
    "Accuracy": [xgb_accuracy, lgbm_accuracy],
    "Training Time (s)": [xgb_time, lgbm_time]
}

results_df = pd.DataFrame(results)
print(results_df)

# Generate classification reports
xgb_report = classification_report(y_test, y_pred_xgb)
lgbm_report = classification_report(y_test, y_pred_lgbm)

# Print classification reports
print("\nClassification Report - XGBoost:\n", xgb_report)
print("\nClassification Report - LGBM:\n", lgbm_report)

# Print confusion matrices
print("\nConfusion Matrix - XGBoost:\n", xgb_conf_matrix)
print("\nConfusion Matrix - LGBM:\n", lgbm_conf_matrix)

# Compute confusion matrices
xgb_conf_matrix = confusion_matrix(y_test, y_pred_xgb)
lgbm_conf_matrix = confusion_matrix(y_test, y_pred_lgbm)

# Function to plot confusion matrices
def plot_conf_matrix(conf_matrix, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Plot for XGBoost
plot_conf_matrix(xgb_conf_matrix, "XGBoost")

# Plot for LightGBM
plot_conf_matrix(lgbm_conf_matrix, "LightGBM")



#2D Scatter Plot using PCA
# Apply PCA to reduce features to 2D
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Create scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette="viridis", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D Scatter Plot of Classes (PCA)")
plt.legend(title="Class")
plt.show()


#3D Scatter Plot using PCA
# Apply PCA to reduce features to 3D
pca_3d = PCA(n_components=3)
X_test_pca_3d = pca_3d.fit_transform(X_test)

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(X_test_pca_3d[:, 0], X_test_pca_3d[:, 1], X_test_pca_3d[:, 2], c=y_test, cmap="viridis", alpha=0.7)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.title("3D Scatter Plot of Classes (PCA)")
plt.colorbar(sc, label="Class")
plt.show()






































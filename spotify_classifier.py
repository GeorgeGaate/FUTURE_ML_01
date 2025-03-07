# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import shap
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score,classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve, f1_score, auc, roc_curve, roc_auc_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import XGBClassifier
import lightgbm as lgb


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

df.info()

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


#Data visualization
# Explicit - Bar Plot
plt.figure(figsize=(10, 5))
sns.countplot(x=df["explicit"], palette="coolwarm")
plt.title("Explicit Songs Count")
plt.xlabel("Explicit")
plt.ylabel("Count")
plt.show()

# Explicit - Pie Chart
plt.figure(figsize=(6, 6))
df["explicit"].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "salmon"])
plt.title("Explicit Songs Proportion")
plt.ylabel("")
plt.show()

# Mode - Bar Plot
plt.figure(figsize=(6, 5))
sns.countplot(x=df["mode"], palette="coolwarm")
plt.title("Mode Distribution")
plt.xlabel("Mode")
plt.ylabel("Count")
plt.show()



# Mode - Pie Chart
plt.figure(figsize=(6, 6))
df["mode"].value_counts().plot.pie(autopct="%1.1f%%", colors=["skyblue", "salmon"])
plt.title("Mode Proportion")
plt.ylabel("")
plt.show()




# Key - Bar Plot
plt.figure(figsize=(10, 5))
sns.countplot(x=df["key"], palette="viridis")
plt.title("Key Distribution")
plt.xlabel("Musical Key")
plt.ylabel("Count")
plt.show()

# Time Signature - Bar Plot
plt.figure(figsize=(8, 5))
sns.countplot(x=df["time_signature"], palette="magma")
plt.title("Time Signature Distribution")
plt.xlabel("Time Signature")
plt.ylabel("Count")
plt.show()

# Track Genre - Bar Plot (Top 50 Genres)
top_genres = df["track_genre"].value_counts().nlargest(50)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette="plasma")
plt.title("Top 50 Most Common Track Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()


# Numerical feature distributions (Histograms & Box Plots)
numerical_features = [
    "popularity", "duration_min", "danceability", "energy",
    "loudness", "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

for feature in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # Histogram
    sns.histplot(df[feature], bins=50, kde=True, ax=axes[0], color="blue")
    axes[0].set_title(f"{feature} Distribution")
    # Box plot
    sns.boxplot(x=df[feature], ax=axes[1], color="red")
    axes[1].set_title(f"{feature} Box Plot")
    # Show the plots
    plt.tight_layout()
    plt.show()




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

#handling outliers
df_numeric = df.select_dtypes(include=['float64', 'int64'])

#outlier detection using boxplots
plt.figure(figsize=(12, 6))
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


# Apply RobustScaler 
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Check if outliers persist after scaling
z_scores_scaled = np.abs(zscore(X_scaled))  
outliers_scaled = (z_scores_scaled > 5).sum()  
print(f"Number of outliers per feature after Robust Scaling:\n{outliers_scaled}")

#Feature Selection with Recursive Feature Elimination (RFE)
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
rfe = RFE(estimator=xgb, n_features_to_select=5)  # Select top 5 features
rfe.fit(X_scaled, y)

# Get selected features
selected_features = X_scaled.columns[rfe.support_]
print("Selected Features:", selected_features)


# Compute correlation matrix
corr_matrix = X_scaled.corr()

# Heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Remove features with high correlation (threshold = 0.85)
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

# Drop redundant features
X_selected = X_scaled.drop(columns=high_corr_features)
print("Features kept after correlation analysis:", X_selected.columns.tolist())



# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_resampled))



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_resampled  # Ensures balanced class distribution
)



# Define models
models = {
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42)
}



# Train & Evaluate each model
for name, model in models.items():
    start_time = time.time()  # Start time
    model.fit(X_train, y_train)
    end_time = time.time()  # End time
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{name} Model Performance:")
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


#Confusion Matrix Heatmap
plt.figure(figsize=(15, 5))

# Loop through models and plot Confusion Matrices
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as a heatmap
    plt.subplot(1, 3, i+1)  # 1 row, 3 columns (for 3 models)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le_mood.classes_, yticklabels=le_mood.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {name}")

plt.tight_layout()
plt.show()



# Define a color map for the models
colors = {
    "XGBoost": "green",
    "LightGBM": "red"
}

#Compute misclassification rates per mood
# Initialize a dictionary to store misclassification results
misclassification_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Calculate misclassification rate for each mood
    total_samples_per_class = cm.sum(axis=1)  # Total true samples per class
    misclassified_per_class = total_samples_per_class - np.diag(cm)  # Off-diagonal elements
    misclassification_rate = misclassified_per_class / total_samples_per_class
    misclassification_results[name] = misclassification_rate
    # Print misclassification results
    print(f"\nMisclassification Rate for {name}:")
    for i, mood in enumerate(le_mood.classes_):
        print(f"{mood}: {misclassification_rate[i]:.2%}")


#visualize misclassification rates per model
plt.figure(figsize=(12, 6))

for name, misclassification_rate in misclassification_results.items():
    plt.plot(le_mood.classes_, misclassification_rate, marker='o', label=name)

plt.xlabel("Mood")
plt.ylabel("Misclassification Rate")
plt.title("Misclassification Rate by Mood for Each Model")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()


# Initialize SHAP explainer for XGBoost
xgb_explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = xgb_explainer(X_test)


# Summary Plot - Feature Importance
shap.summary_plot(shap_values, X_test)


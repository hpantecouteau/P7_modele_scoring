import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("full_clean_dataset.csv")
print(df.head())
features = [col for col in df.columns if col != "TARGET" and col != "SK_ID_CURR"]
X_train, X_test, y_train, y_test = train_test_split(df[features], df.TARGET, test_size=0.2, stratify=df.TARGET, random_state=0)
print(type(X_train))
print(y_train)
import pandas as pd

df_input = pd.read_csv("input.csv")
df_shap = pd.read_csv("df_shap.csv")
print(f"df_input : {df_input.shape}")
print(f"df_shap : {df_shap.shape}")
sampled_df_shap = df_shap.sample(n=500, random_state=0)
sampled_df_input = df_input.loc[df_input.SK_ID_CURR.isin(sampled_df_shap.SK_ID_CURR.values),:]
print(f"sampled_df_shap : {sampled_df_shap.shape}")
print(f"sampled_df_input : {sampled_df_input.shape}")
print("Recording in new CSV files...")
sampled_df_input.to_csv("sampled_df_input.csv", index=False)
sampled_df_shap.to_csv("sampled_df_shap.csv", index=False)
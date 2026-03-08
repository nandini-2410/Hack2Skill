import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===============================
# LOAD DATA
# ===============================

data = pd.read_csv("Cleaned_Dataset.csv")

# ===============================
# BASIC CLEANING
# ===============================

data = data.fillna(method="ffill").fillna(method="bfill")

# ===============================
# ORGAN FAILURE FEATURES
# ===============================

data["Respiratory_Failure"] = ((data["O2Sat"] < 90) | (data["Resp"] > 22)).astype(int)

data["Circulatory_Failure"] = (
    (data["MAP"] < 65) |
    (data["SBP"] < 90) |
    (data["HR"] > 100)
).astype(int)

data["Kidney_Failure"] = (data["Creatinine"] > 1.5).astype(int)
data["Liver_Failure"] = (data["Bilirubin_total"] > 2).astype(int)
data["Coagulation_Failure"] = (data["Platelets"] < 150).astype(int)

data["Infection_Shock"] = (
    (data["WBC"] > 12000) |
    (data["WBC"] < 4000) |
    (data["Temp"] > 38) |
    (data["Temp"] < 36) |
    (data["Lactate"] > 2)
).astype(int)

data["Multi_Organ_Failure_Score"] = data[[
    "Respiratory_Failure",
    "Circulatory_Failure",
    "Kidney_Failure",
    "Liver_Failure",
    "Coagulation_Failure"
]].sum(axis=1)

# ===============================
# SOFA SCORE
# ===============================

data["SOFA_Resp"] = np.select(
[
data["O2Sat"] >= 96,
(data["O2Sat"] >= 92) & (data["O2Sat"] < 96),
(data["O2Sat"] >= 88) & (data["O2Sat"] < 92),
(data["O2Sat"] >= 85) & (data["O2Sat"] < 88),
data["O2Sat"] < 85
],
[0,1,2,3,4]
)

data["SOFA_Coag"] = np.select(
[
data["Platelets"] >= 150,
(data["Platelets"] >= 100) & (data["Platelets"] < 150),
(data["Platelets"] >= 50) & (data["Platelets"] < 100),
(data["Platelets"] >= 20) & (data["Platelets"] < 50),
data["Platelets"] < 20
],
[0,1,2,3,4]
)

data["SOFA_Liver"] = np.select(
[
data["Bilirubin_total"] < 1.2,
(data["Bilirubin_total"] >= 1.2) & (data["Bilirubin_total"] < 2),
(data["Bilirubin_total"] >= 2) & (data["Bilirubin_total"] < 6),
(data["Bilirubin_total"] >= 6) & (data["Bilirubin_total"] < 12),
data["Bilirubin_total"] >= 12
],
[0,1,2,3,4]
)

data["SOFA_Cardio"] = np.where(data["MAP"] < 70,1,0)

data["SOFA_Kidney"] = np.select(
[
data["Creatinine"] < 1.2,
(data["Creatinine"] >= 1.2) & (data["Creatinine"] < 2),
(data["Creatinine"] >= 2) & (data["Creatinine"] < 3.5),
(data["Creatinine"] >= 3.5) & (data["Creatinine"] < 5),
data["Creatinine"] >= 5
],
[0,1,2,3,4]
)

data["SOFA_SCORE"] = (

data["SOFA_Resp"] +
data["SOFA_Coag"] +
data["SOFA_Liver"] +
data["SOFA_Cardio"] +
data["SOFA_Kidney"]

)

# ===============================
# qSOFA SCORE
# ===============================

data["qSOFA_Resp"] = (data["Resp"] > 22).astype(int)
data["qSOFA_BP"] = (data["SBP"] < 100).astype(int)

data["qSOFA_SCORE"] = (
data["qSOFA_Resp"] +
data["qSOFA_BP"]
)

# ===============================
# ORGAN DETERIORATION TRENDS
# ===============================

data["HR_trend"] = data["HR"].diff().fillna(0)
data["O2Sat_trend"] = data["O2Sat"].diff().fillna(0)
data["MAP_trend"] = data["MAP"].diff().fillna(0)
data["Creatinine_trend"] = data["Creatinine"].diff().fillna(0)
data["Lactate_trend"] = data["Lactate"].diff().fillna(0)

# ===============================
# AGE AWARE FEATURES
# ===============================

data["Age_HR_Risk"] = data["Age"] * data["HR"]
data["Age_Creatinine_Risk"] = data["Age"] * data["Creatinine"]
data["Age_Lactate_Risk"] = data["Age"] * data["Lactate"]

# ===============================
# FEATURE LIST
# ===============================

FEATURES = [

"Hour","HR","O2Sat","Temp","SBP","MAP","Resp",
"Lactate","Creatinine","Platelets","Bilirubin_total",
"WBC","Age","Gender",

"HR_trend","O2Sat_trend","MAP_trend",
"Creatinine_trend","Lactate_trend",

"Respiratory_Failure","Circulatory_Failure","Kidney_Failure",
"Liver_Failure","Coagulation_Failure","Infection_Shock",
"Multi_Organ_Failure_Score",

"SOFA_SCORE",
"qSOFA_SCORE",

"Age_HR_Risk","Age_Creatinine_Risk","Age_Lactate_Risk"

]

TARGET = "SepsisLabel"

X = data[FEATURES]
y = data[TARGET]

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)

# ===============================
# SCALING
# ===============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler,open("scaler.pkl","wb"))

# ===============================
# LSTM INPUT
# ===============================

X_train_lstm = X_train_scaled.reshape(
X_train_scaled.shape[0],1,X_train_scaled.shape[1]
)

X_test_lstm = X_test_scaled.reshape(
X_test_scaled.shape[0],1,X_test_scaled.shape[1]
)

# ===============================
# LSTM MODEL
# ===============================

lstm_model = Sequential()

lstm_model.add(LSTM(64,input_shape=(1,X_train_scaled.shape[1])))

lstm_model.add(Dropout(0.3))

lstm_model.add(Dense(32,activation="relu"))

lstm_model.add(Dense(1,activation="sigmoid"))

lstm_model.compile(
loss="binary_crossentropy",
optimizer="adam",
metrics=["accuracy"]
)

lstm_model.fit(
X_train_lstm,
y_train,
epochs=10,
batch_size=32,
validation_split=0.2
)

lstm_model.save("lstm_model.h5")

# ===============================
# LSTM FEATURE EXTRACTOR
# ===============================

feature_extractor = Model(
inputs=lstm_model.inputs,
outputs=lstm_model.layers[-2].output
)

lstm_train_features = feature_extractor.predict(X_train_lstm)
lstm_test_features = feature_extractor.predict(X_test_lstm)

# ===============================
# XGBOOST MODEL
# ===============================

xgb_model = XGBClassifier(

n_estimators=300,
max_depth=6,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
eval_metric="logloss"

)

xgb_model.fit(lstm_train_features,y_train)

pickle.dump(xgb_model,open("xgb_model.pkl","wb"))

# ===============================
# MODEL EVALUATION
# ===============================

preds = xgb_model.predict(lstm_test_features)

accuracy = (preds == y_test).mean()

print("\nModel Accuracy:",accuracy)

# ===============================
# SHAP EXPLAINABILITY
# ===============================

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(lstm_test_features)

shap.summary_plot(
shap_values,
lstm_test_features,
show=False
)

plt.savefig("static/shap_summary.png")

print("SHAP explanation saved to static/shap_summary.png")
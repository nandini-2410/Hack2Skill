from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model, Model

app = Flask(__name__)



scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lstm_model = load_model("lstm_model.h5")

# LSTM feature extractor
feature_extractor = Model(
    inputs=lstm_model.inputs,
    outputs=lstm_model.layers[-2].output
)



FEATURES = [

"Hour","HR","O2Sat","Temp","SBP","MAP","Resp",
"Lactate","Creatinine","Platelets","Bilirubin_total",
"WBC","Age","Gender",

"HR_trend","O2Sat_trend","MAP_trend",
"Creatinine_trend","Lactate_trend",

"Respiratory_Failure","Circulatory_Failure",
"Kidney_Failure","Liver_Failure",
"Coagulation_Failure","Infection_Shock",
"Multi_Organ_Failure_Score",

"SOFA_SCORE",
"qSOFA_SCORE",

"Age_HR_Risk","Age_Creatinine_Risk","Age_Lactate_Risk"

]



@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    df = pd.DataFrame([data])

    

    df["Respiratory_Failure"] = ((df["O2Sat"] < 90) | (df["Resp"] > 22)).astype(int)

    df["Circulatory_Failure"] = (
        (df["MAP"] < 65) |
        (df["SBP"] < 90) |
        (df["HR"] > 100)
    ).astype(int)

    df["Kidney_Failure"] = (df["Creatinine"] > 1.5).astype(int)
    df["Liver_Failure"] = (df["Bilirubin_total"] > 2).astype(int)
    df["Coagulation_Failure"] = (df["Platelets"] < 150).astype(int)

    df["Infection_Shock"] = (
        (df["WBC"] > 12000) |
        (df["WBC"] < 4000) |
        (df["Temp"] > 38) |
        (df["Temp"] < 36) |
        (df["Lactate"] > 2)
    ).astype(int)

    df["Multi_Organ_Failure_Score"] = (

        df["Respiratory_Failure"] +
        df["Circulatory_Failure"] +
        df["Kidney_Failure"] +
        df["Liver_Failure"] +
        df["Coagulation_Failure"]

    )

    

    df["SOFA_Resp"] = np.select(
        [
            df["O2Sat"] >= 96,
            (df["O2Sat"] >= 92) & (df["O2Sat"] < 96),
            (df["O2Sat"] >= 88) & (df["O2Sat"] < 92),
            (df["O2Sat"] >= 85) & (df["O2Sat"] < 88),
            df["O2Sat"] < 85
        ],
        [0,1,2,3,4],
        default=0
    )

    df["SOFA_Coag"] = np.select(
        [
            df["Platelets"] >= 150,
            (df["Platelets"] >= 100) & (df["Platelets"] < 150),
            (df["Platelets"] >= 50) & (df["Platelets"] < 100),
            (df["Platelets"] >= 20) & (df["Platelets"] < 50),
            df["Platelets"] < 20
        ],
        [0,1,2,3,4],
        default=0
    )

    df["SOFA_Liver"] = np.select(
        [
            df["Bilirubin_total"] < 1.2,
            (df["Bilirubin_total"] >= 1.2) & (df["Bilirubin_total"] < 2),
            (df["Bilirubin_total"] >= 2) & (df["Bilirubin_total"] < 6),
            (df["Bilirubin_total"] >= 6) & (df["Bilirubin_total"] < 12),
            df["Bilirubin_total"] >= 12
        ],
        [0,1,2,3,4],
        default=0
    )

    df["SOFA_Cardio"] = np.where(df["MAP"] < 70, 1, 0)

    df["SOFA_Kidney"] = np.select(
        [
            df["Creatinine"] < 1.2,
            (df["Creatinine"] >= 1.2) & (df["Creatinine"] < 2),
            (df["Creatinine"] >= 2) & (df["Creatinine"] < 3.5),
            (df["Creatinine"] >= 3.5) & (df["Creatinine"] < 5),
            df["Creatinine"] >= 5
        ],
        [0,1,2,3,4],
        default=0
    )

    df["SOFA_SCORE"] = (

        df["SOFA_Resp"] +
        df["SOFA_Coag"] +
        df["SOFA_Liver"] +
        df["SOFA_Cardio"] +
        df["SOFA_Kidney"]

    )

   

    df["qSOFA_Resp"] = (df["Resp"] > 22).astype(int)
    df["qSOFA_BP"] = (df["SBP"] < 100).astype(int)

    df["qSOFA_SCORE"] = (
        df["qSOFA_Resp"] +
        df["qSOFA_BP"]
    )

   

    df["HR_trend"] = 0
    df["O2Sat_trend"] = 0
    df["MAP_trend"] = 0
    df["Creatinine_trend"] = 0
    df["Lactate_trend"] = 0

    # ===============================
    # AGE RISK FEATURES
    # ===============================

    df["Age_HR_Risk"] = df["Age"] * df["HR"]
    df["Age_Creatinine_Risk"] = df["Age"] * df["Creatinine"]
    df["Age_Lactate_Risk"] = df["Age"] * df["Lactate"]

    # ===============================
    # MODEL INPUT
    # ===============================

    X = df[FEATURES]

    X_scaled = scaler.transform(X)

    X_lstm = X_scaled.reshape(
        X_scaled.shape[0],
        1,
        X_scaled.shape[1]
    )

    lstm_features = feature_extractor.predict(X_lstm)

    prob = xgb_model.predict_proba(lstm_features)[:,1]

    risk_score = float(prob[0])

    if risk_score > 0.7:
        alert = "HIGH"
    elif risk_score > 0.5:
        alert = "MODERATE"
    else:
        alert = "LOW"

    pred = int(risk_score >= 0.55)

    return jsonify({

        "sepsis_prediction": pred,
        "risk_probability": risk_score,
        "risk_percent": risk_score * 100,
        "alert_level": alert,
        "SOFA_SCORE": int(df["SOFA_SCORE"].iloc[0]),
        "qSOFA_SCORE": int(df["qSOFA_SCORE"].iloc[0])

    })


# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
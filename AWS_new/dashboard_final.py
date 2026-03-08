import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import os
import requests

app = Flask(__name__)

DATA_FILE = "patient_vitals.csv"



def create_line_chart(df, column, filename):

    df = df.tail(100)
    df = df.reset_index(drop=True)
    df["Minute"] = df.index

    plt.figure(figsize=(4,3))
    plt.plot(df["Minute"], df[column], color="cyan", linewidth=2)

    plt.xlabel("Minute")
    plt.ylabel(column)
    plt.title(column)
    plt.grid(True)

    chart_path = os.path.join("static", filename)

    plt.savefig(chart_path)
    plt.close()

    return chart_path



def create_failure_donut(value, title, filename):

    labels = ["Normal", "Failure"]
    values = [1 - value, value]

    plt.figure(figsize=(3,3))

    plt.pie(
        values,
        labels=labels,
        autopct='%1.0f%%',
        colors=["green","red"],
        wedgeprops={'width':0.4}
    )

    plt.title(title)

    chart_path = os.path.join("static", filename)

    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    return chart_path



@app.route("/")
def dashboard():

    df = pd.read_csv(DATA_FILE)

    patient_ids = df["patient_id"].unique()

    selected_patient = request.args.get("patient_id")

    if selected_patient is None:
        selected_patient = patient_ids[0]

    selected_patient = (selected_patient)

    patient_data = df[df["patient_id"] == selected_patient]

    latest = patient_data.iloc[-1]

    gender_text = "Male" if latest["Gender"] == 1 else "Female"


    

    api_data = {
        "Hour": int(latest["Hour"]),
        "HR": float(latest["HR"]),
        "O2Sat": float(latest["O2Sat"]),
        "Temp": float(latest["Temp"]),
        "SBP": float(latest["SBP"]),
        "MAP": float(latest["MAP"]),
        "Resp": float(latest["Resp"]),
        "Lactate": float(latest["Lactate"]),
        "Creatinine": float(latest["Creatinine"]),
        "Platelets": float(latest["Platelets"]),
        "Bilirubin_total": float(latest["Bilirubin_total"]),
        "WBC": float(latest["WBC"]),
        "Age": int(latest["Age"]),
        "Gender": int(latest["Gender"])
    }

    try:

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=api_data,
            timeout=5
        )

        result = response.json()

        sepsis_prediction = result.get("sepsis_prediction",0)
        sepsis_risk = result.get("risk_probability",0)
        risk_percent = result.get("risk_percent",0)
        alert_level = result.get("alert_level","UNKNOWN")

        sofa_score = result.get("SOFA_SCORE",0)
        qsofa_score = result.get("qSOFA_SCORE",0)

    except Exception as e:

        print("API ERROR:", e)

        sepsis_prediction = "API ERROR"
        sepsis_risk = 0
        risk_percent = 0
        alert_level = "UNKNOWN"
        sofa_score = 0
        qsofa_score = 0


  

    charts = {}

    vitals = [
        "HR","O2Sat","Temp","MAP","Resp",
        "Lactate","Creatinine","Platelets",
        "Bilirubin_total","WBC"
    ]

    for v in vitals:
        charts[v] = create_line_chart(patient_data, v, f"{v}.png")


    

    resp_fail_chart = create_failure_donut(
        latest.get("Respiratory_Failure",0),
        "Respiratory Failure",
        "resp.png"
    )

    circ_fail_chart = create_failure_donut(
        latest.get("Circulatory_Failure",0),
        "Circulatory Failure",
        "circ.png"
    )

    kidney_fail_chart = create_failure_donut(
        latest.get("Kidney_Failure",0),
        "Kidney Failure",
        "kidney.png"
    )

    liver_fail_chart = create_failure_donut(
        latest.get("Liver_Failure",0),
        "Liver Failure",
        "liver.png"
    )

    coag_fail_chart = create_failure_donut(
        latest.get("Coagulation_Failure",0),
        "Coagulation Failure",
        "coag.png"
    )

    shock_chart = create_failure_donut(
        latest.get("Infection_Shock",0),
        "Infection Shock",
        "shock.png"
    )


  

    return render_template(

        "dashboard.html",

        data=latest,
        gender_text=gender_text,

        sepsis_prediction=sepsis_prediction,
        sepsis_risk=sepsis_risk,
        risk_percent=risk_percent,
        alert_level=alert_level,

        sofa_score=sofa_score,
        qsofa_score=qsofa_score,

        patient_ids=patient_ids,
        selected_patient=selected_patient,

        charts=charts,

        resp_fail_chart=resp_fail_chart,
        circ_fail_chart=circ_fail_chart,
        kidney_fail_chart=kidney_fail_chart,
        liver_fail_chart=liver_fail_chart,
        coag_fail_chart=coag_fail_chart,
        shock_chart=shock_chart
    )



if __name__ == "__main__":
    app.run(debug=True, port=5001)
import requests

url = "http://127.0.0.1:5000/predict"

data = {
"Hour":5,
"HR":110,
"O2Sat":85,
"Temp":39,
"SBP":85,
"MAP":60,
"Resp":24,
"Lactate":3,
"Creatinine":1.8,
"Platelets":120,
"Bilirubin_total":2.5,
"WBC":14,
"Age":65,
"Gender":"M"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response Text:", response.text)
import csv
import os
import random
from datetime import datetime, timedelta
import time
import threading



VITALS_FILE = "patient_vitals.csv"
PATIENT_FILE = "patients.csv"

PATIENT_LIMIT = 8
INTERVAL_SECONDS = 60   # change to 5 for faster testing

SEPSIS_TRIGGER_MINUTES = 2  # SEPSIS STARTS AFTER 2 MINUTES



last_temperature = {}
start_time = datetime.now()

patients = []
critical_patients = []



def initialize_files():

    if not os.path.exists(PATIENT_FILE):

        with open(PATIENT_FILE,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id","admitted_at","status"])

    if not os.path.exists(VITALS_FILE):

        with open(VITALS_FILE,'w',newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                "patient_id",
                "timestamp",
                "Hour",
                "HR",
                "O2Sat",
                "Temp",
                "SBP",
                "MAP",
                "Resp",
                "Lactate",
                "Creatinine",
                "Platelets",
                "Bilirubin_total",
                "WBC",
                "Age",
                "Gender",
                "Respiratory_Failure",
                "Circulatory_Failure",
                "Kidney_Failure",
                "Liver_Failure",
                "Coagulation_Failure",
                "Infection_Shock",
                "Multi_Organ_Failure_Score"
            ])


def load_patients():

    global patients
    patients = []

    with open(PATIENT_FILE,'r') as f:

        reader = csv.DictReader(f)

        for row in reader:
            if row["status"] == "active":
                patients.append(row["patient_id"])

# ==============================
# ADD PATIENT
# ==============================

def add_patient():

    new_id = f"P{str(len(patients)+1).zfill(3)}"

    with open(PATIENT_FILE,'a',newline='') as f:

        writer = csv.writer(f)
        writer.writerow([new_id,datetime.now(),"active"])

    patients.append(new_id)

    print("New patient added:",new_id)

# ==============================
# NORMAL VITAL GENERATORS
# ==============================

def normal_heart_rate():
    return random.randint(60,90)

def normal_spo2():
    return random.randint(96,100)

def normal_bp():
    return random.randint(110,130),random.randint(70,85)

def normal_temperature():
    return round(random.uniform(36.5,37.5),1)

def normal_resp():
    return random.randint(12,18)

def normal_lactate():
    return round(random.uniform(0.5,1.8),2)

def normal_creatinine():
    return round(random.uniform(0.6,1.2),2)

def normal_platelets():
    return random.randint(180,350)

def normal_bilirubin():
    return round(random.uniform(0.3,1.0),2)

def normal_wbc():
    return random.randint(4000,11000)

def patient_age():
    return random.randint(25,80)

def patient_gender():
    return random.choice([0,1])



def critical_heart_rate(minutes_since):

    value = 95 + (minutes_since // 2)
    return min(value,130)

def critical_bp():
    return random.randint(70,89),random.randint(50,64)

def critical_spo2():
    return random.randint(88,94)

def critical_temperature(minutes_since):

    value = 37.8 + (minutes_since*0.03)
    return round(min(value,40),1)

def critical_resp():
    return random.randint(22,35)

def critical_lactate():
    return round(random.uniform(2.5,6.5),2)

def critical_creatinine():
    return round(random.uniform(1.5,3.5),2)

def critical_platelets():
    return random.randint(50,150)

def critical_bilirubin():
    return round(random.uniform(1.5,5.0),2)

def critical_wbc():
    return random.randint(12000,25000)

# ==============================
# MAP CALCULATION
# ==============================

def calculate_map(sys,dia):
    return round((sys + 2*dia)/3,2)



def organ_failure(resp,map_val,creat,bili,platelets):

    respiratory = 1 if resp > 22 else 0
    circulatory = 1 if map_val < 65 else 0
    kidney = 1 if creat > 2 else 0
    liver = 1 if bili > 2 else 0
    coag = 1 if platelets < 100 else 0

    score = respiratory + circulatory + kidney + liver + coag

    shock = 1 if score >= 3 else 0

    return respiratory,circulatory,kidney,liver,coag,shock,score



def get_retention_hours():

    if len(patients) > PATIENT_LIMIT:
        return 12

    return 24



def append_reading(patient_id,timestamp):

    elapsed = timestamp - start_time
    elapsed_minutes = int(elapsed.total_seconds()/60)

    hour = elapsed_minutes

    is_critical = (
        patient_id in critical_patients and elapsed_minutes >= SEPSIS_TRIGGER_MINUTES
    )

    if is_critical:

        mins = elapsed_minutes - SEPSIS_TRIGGER_MINUTES

        hr = critical_heart_rate(mins)
        spo2 = critical_spo2()
        sys_bp,dia_bp = critical_bp()

        resp = critical_resp()
        lactate = critical_lactate()
        creat = critical_creatinine()
        platelets = critical_platelets()
        bilirubin = critical_bilirubin()
        wbc = critical_wbc()

        if timestamp.minute % 30 == 0 or patient_id not in last_temperature:

            temp = critical_temperature(mins)
            last_temperature[patient_id] = temp
        else:
            temp = last_temperature[patient_id]

    else:

        hr = normal_heart_rate()
        spo2 = normal_spo2()
        sys_bp,dia_bp = normal_bp()

        resp = normal_resp()
        lactate = normal_lactate()
        creat = normal_creatinine()
        platelets = normal_platelets()
        bilirubin = normal_bilirubin()
        wbc = normal_wbc()

        if timestamp.minute % 30 == 0 or patient_id not in last_temperature:

            temp = normal_temperature()
            last_temperature[patient_id] = temp
        else:
            temp = last_temperature[patient_id]

    age = patient_age()
    gender = patient_gender()

    map_val = calculate_map(sys_bp,dia_bp)

    resp_fail,circ_fail,kidney_fail,liver_fail,coag_fail,shock,score = organ_failure(
        resp,map_val,creat,bilirubin,platelets
    )

    with open(VITALS_FILE,'a',newline='') as f:

        writer = csv.writer(f)

        writer.writerow([
            patient_id,
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            hour,
            hr,
            spo2,
            temp,
            sys_bp,
            map_val,
            resp,
            lactate,
            creat,
            platelets,
            bilirubin,
            wbc,
            age,
            gender,
            resp_fail,
            circ_fail,
            kidney_fail,
            liver_fail,
            coag_fail,
            shock,
            score
        ])



def prune_old_data():

    retention = get_retention_hours()

    cutoff = datetime.now() - timedelta(hours=retention)

    rows = []

    with open(VITALS_FILE,'r') as f:

        reader = csv.reader(f)
        header = next(reader)

        for row in reader:

            row_time = datetime.strptime(row[1],"%Y-%m-%d %H:%M:%S")

            if row_time >= cutoff:
                rows.append(row)

    with open(VITALS_FILE,'w',newline='') as f:

        writer = csv.writer(f)

        writer.writerow(header)
        writer.writerows(rows)



def monitoring_loop():

    while True:

        now = datetime.now()

        for patient in patients:
            append_reading(patient,now)

        prune_old_data()

        print(
            "Updated:",
            now.strftime("%H:%M:%S"),
            "| Patients:",
            len(patients),
            "| Critical:",
            critical_patients
        )

        time.sleep(INTERVAL_SECONDS)


initialize_files()

# Create patients if file empty
if os.stat(PATIENT_FILE).st_size <= 40:

    with open(PATIENT_FILE,'a',newline='') as f:

        writer = csv.writer(f)

        for i in range(1,11):

            pid = f"P{str(i).zfill(3)}"
            writer.writerow([pid,datetime.now(),"active"])

load_patients()

# Choose 40% critical patients
critical_count = max(4,int(len(patients)*0.4))
critical_patients = random.sample(patients,critical_count)

print("Monitoring started...")
print("Critical patients:",critical_patients)

t1 = threading.Thread(target=monitoring_loop)
t1.start()
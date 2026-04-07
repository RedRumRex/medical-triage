"""
Patient case library — all cases derived from real dataset symptom distributions.
Each patient is crafted with authentic symptom profiles from the
246,945-record Augmented Disease & Symptoms dataset.
"""
from __future__ import annotations
from .models import Patient, Vitals

# ─── EASY TASK: 3 patients, textbook presentations ───────────────────────────
EASY_PATIENTS: list[Patient] = [
    Patient(
        patient_id="E01",
        age=62, sex="M",
        chief_complaint="Crushing chest pain radiating to left arm, diaphoresis",
        symptoms=[
            "burning chest pain", "arm pain", "shortness of breath",
            "sweating", "nausea", "irregular heartbeat", "fainting",
            "increased heart rate", "chest tightness"
        ],
        vitals=Vitals(
            heart_rate=112, blood_pressure_sys=88, blood_pressure_dia=60,
            respiratory_rate=22, oxygen_saturation=93.0, temperature=36.8, gcs=14
        ),
        history="Hypertension, hyperlipidemia. Smokes 1 ppd x 30 years.",
        true_disease="heart attack", true_esi=1,
    ),
    Patient(
        patient_id="E02",
        age=34, sex="F",
        chief_complaint="Dysuria, urinary frequency, suprapubic pain x 2 days",
        symptoms=[
            "painful urination", "frequent urination", "suprapubic pain",
            "blood in urine", "lower abdominal pain", "back pain"
        ],
        vitals=Vitals(
            heart_rate=84, blood_pressure_sys=118, blood_pressure_dia=76,
            respiratory_rate=16, oxygen_saturation=98.5, temperature=37.2, gcs=15
        ),
        history="No significant history. Sexually active. Last UTI 2 years ago.",
        true_disease="urinary tract infection", true_esi=3,
    ),
    Patient(
        patient_id="E03",
        age=27, sex="M",
        chief_complaint="Palpitations, feeling anxious, breathing fast",
        symptoms=[
            "palpitations", "anxiety and nervousness", "breathing fast",
            "chest tightness", "dizziness", "insomnia", "depression",
            "depressive or psychotic symptoms"
        ],
        vitals=Vitals(
            heart_rate=96, blood_pressure_sys=128, blood_pressure_dia=82,
            respiratory_rate=20, oxygen_saturation=99.0, temperature=36.6, gcs=15
        ),
        history="No cardiac history. Reports high job stress. No medications.",
        true_disease="panic disorder", true_esi=5,
    ),
]

# ─── MEDIUM TASK: 8 patients, overlapping severity ───────────────────────────
MEDIUM_PATIENTS: list[Patient] = [
    Patient(
        patient_id="M01",
        age=71, sex="F",
        chief_complaint="Sudden-onset left-sided weakness and slurred speech",
        symptoms=[
            "focal weakness", "slurring words", "headache", "problems with movement",
            "dizziness", "difficulty speaking", "disturbance of memory",
            "abnormal involuntary movements"
        ],
        vitals=Vitals(
            heart_rate=88, blood_pressure_sys=192, blood_pressure_dia=110,
            respiratory_rate=18, oxygen_saturation=95.0, temperature=37.0, gcs=13
        ),
        history="Atrial fibrillation on warfarin. Hypertension. Onset ~45 min ago.",
        true_disease="stroke", true_esi=1,
    ),
    Patient(
        patient_id="M02",
        age=45, sex="M",
        chief_complaint="Fever, confusion, severe headache, neck stiffness",
        symptoms=[
            "fever", "headache", "neck stiffness or tightness", "neck pain",
            "nausea", "vomiting", "ache all over", "cough",
            "depressive or psychotic symptoms"
        ],
        vitals=Vitals(
            heart_rate=118, blood_pressure_sys=98, blood_pressure_dia=62,
            respiratory_rate=24, oxygen_saturation=94.0, temperature=39.8, gcs=12
        ),
        history="No significant PMH. No recent travel.",
        true_disease="meningitis", true_esi=1,
    ),
    Patient(
        patient_id="M03",
        age=55, sex="M",
        chief_complaint="Right lower quadrant pain, anorexia, low-grade fever x 18h",
        symptoms=[
            "lower abdominal pain", "decreased appetite", "fever",
            "nausea", "vomiting", "side pain", "sharp abdominal pain",
            "stomach bloating", "upper abdominal pain"
        ],
        vitals=Vitals(
            heart_rate=98, blood_pressure_sys=124, blood_pressure_dia=78,
            respiratory_rate=18, oxygen_saturation=97.0, temperature=38.3, gcs=15
        ),
        history="No abdominal surgeries. No medications.",
        true_disease="appendicitis", true_esi=2,
    ),
    Patient(
        patient_id="M04",
        age=38, sex="F",
        chief_complaint="Pleuritic chest pain, leg swelling after long flight",
        symptoms=[
            "sharp chest pain", "leg pain", "leg swelling", "shortness of breath",
            "difficulty breathing", "sweating", "cough", "hemoptysis",
            "side pain", "weakness"
        ],
        vitals=Vitals(
            heart_rate=108, blood_pressure_sys=102, blood_pressure_dia=68,
            respiratory_rate=26, oxygen_saturation=91.0, temperature=37.5, gcs=15
        ),
        history="OCP use. 14-hour flight 2 days ago. No prior clots.",
        true_disease="pulmonary embolism", true_esi=1,
    ),
    Patient(
        patient_id="M05",
        age=29, sex="F",
        chief_complaint="Wheezing, cough, difficulty breathing — known asthmatic",
        symptoms=[
            "wheezing", "cough", "difficulty breathing", "shortness of breath",
            "chest tightness", "coughing up sputum", "nasal congestion",
            "allergic reaction", "sharp chest pain"
        ],
        vitals=Vitals(
            heart_rate=102, blood_pressure_sys=130, blood_pressure_dia=84,
            respiratory_rate=28, oxygen_saturation=93.0, temperature=37.1, gcs=15
        ),
        history="Asthma since childhood. Used salbutamol inhaler x3 today.",
        true_disease="asthma", true_esi=3,
    ),
    Patient(
        patient_id="M06",
        age=68, sex="M",
        chief_complaint="Productive cough, fever, dyspnea x 3 days",
        symptoms=[
            "cough", "fever", "difficulty breathing", "shortness of breath",
            "sharp chest pain", "weakness", "chills", "vomiting",
            "wheezing", "coryza", "nasal congestion"
        ],
        vitals=Vitals(
            heart_rate=94, blood_pressure_sys=134, blood_pressure_dia=82,
            respiratory_rate=22, oxygen_saturation=92.0, temperature=38.9, gcs=15
        ),
        history="COPD. Smoker 40 pack-years. Lives in nursing facility.",
        true_disease="pneumonia", true_esi=2,
    ),
    Patient(
        patient_id="M07",
        age=42, sex="F",
        chief_complaint="Severe unilateral headache with nausea, photophobia",
        symptoms=[
            "headache", "nausea", "vomiting", "dizziness", "diminished vision",
            "spots or clouds in vision", "blindness", "disturbance of memory",
            "symptoms of the face"
        ],
        vitals=Vitals(
            heart_rate=76, blood_pressure_sys=138, blood_pressure_dia=88,
            respiratory_rate=16, oxygen_saturation=99.0, temperature=36.9, gcs=15
        ),
        history="Migraines since age 22. Took sumatriptan 2h ago — no relief.",
        true_disease="migraine", true_esi=3,
    ),
    Patient(
        patient_id="M08",
        age=22, sex="M",
        chief_complaint="Nausea, vomiting, diarrhea x 24h — possible food poisoning",
        symptoms=[
            "nausea", "vomiting", "diarrhea", "decreased appetite",
            "sharp abdominal pain", "burning abdominal pain", "fever",
            "chills", "fluid retention", "flu-like syndrome"
        ],
        vitals=Vitals(
            heart_rate=88, blood_pressure_sys=116, blood_pressure_dia=74,
            respiratory_rate=16, oxygen_saturation=99.0, temperature=37.8, gcs=15
        ),
        history="Ate at a picnic yesterday. Roommate has similar symptoms.",
        true_disease="infectious gastroenteritis", true_esi=4,
    ),
]

# ─── HARD TASK: 15 patients — misleading vitals, time-sensitive, overlapping ─
HARD_PATIENTS: list[Patient] = [
    Patient(
        patient_id="H01",
        age=78, sex="M",
        chief_complaint="Indigestion, mild back pain — took antacids, no relief",
        symptoms=[
            "burning abdominal pain", "back pain", "palpitations",
            "shortness of breath", "arm swelling", "sharp abdominal pain",
            "dizziness"
        ],
        vitals=Vitals(
            heart_rate=74, blood_pressure_sys=148, blood_pressure_dia=92,
            respiratory_rate=18, oxygen_saturation=96.0, temperature=36.7, gcs=15
        ),
        history="Hypertension, GERD. Vitals look stable — but sudden onset back + abdo pain.",
        true_disease="abdominal aortic aneurysm", true_esi=1,
    ),
    Patient(
        patient_id="H02",
        age=33, sex="F",
        chief_complaint="Sharp lower abdominal pain, missed period, vaginal spotting",
        symptoms=[
            "intermenstrual bleeding", "lower abdominal pain", "pelvic pain",
            "sharp abdominal pain", "dizziness", "nausea", "vaginal discharge",
            "pain during pregnancy"
        ],
        vitals=Vitals(
            heart_rate=106, blood_pressure_sys=96, blood_pressure_dia=58,
            respiratory_rate=20, oxygen_saturation=97.0, temperature=36.9, gcs=15
        ),
        history="LMP 7 weeks ago. Beta-hCG positive. No prior pregnancies.",
        true_disease="ectopic pregnancy", true_esi=1,
    ),
    Patient(
        patient_id="H03",
        age=65, sex="M",
        chief_complaint="Fever, altered mental status, rapid breathing",
        symptoms=[
            "fever", "shortness of breath", "difficulty breathing",
            "feeling ill", "weakness", "vomiting", "cough",
            "decreased appetite", "sharp abdominal pain", "chills"
        ],
        vitals=Vitals(
            heart_rate=124, blood_pressure_sys=86, blood_pressure_dia=54,
            respiratory_rate=28, oxygen_saturation=90.0, temperature=39.6, gcs=11
        ),
        history="Diabetic. Recent UTI treated with antibiotics. Now confused.",
        true_disease="sepsis", true_esi=1,
    ),
    Patient(
        patient_id="H04",
        age=52, sex="M",
        chief_complaint="Palpitations, irregular heartbeat, near-syncope",
        symptoms=[
            "palpitations", "irregular heartbeat", "decreased heart rate",
            "shortness of breath", "weakness", "sharp chest pain",
            "difficulty breathing", "arm swelling", "fainting"
        ],
        vitals=Vitals(
            heart_rate=38, blood_pressure_sys=78, blood_pressure_dia=50,
            respiratory_rate=22, oxygen_saturation=92.0, temperature=36.8, gcs=13
        ),
        history="No cardiac history. Symptoms started 30 min ago.",
        true_disease="cardiac arrest", true_esi=1,
    ),
    Patient(
        patient_id="H05",
        age=44, sex="F",
        chief_complaint="Sudden severe headache — 'worst of my life', photophobia",
        symptoms=[
            "headache", "nausea", "vomiting", "neck pain",
            "neck stiffness or tightness", "fever", "ache all over",
            "depressive or psychotic symptoms"
        ],
        vitals=Vitals(
            heart_rate=90, blood_pressure_sys=158, blood_pressure_dia=96,
            respiratory_rate=18, oxygen_saturation=98.0, temperature=38.1, gcs=14
        ),
        history="No prior headaches. Sudden onset during exertion. LP pending.",
        true_disease="meningitis", true_esi=1,
    ),
    Patient(
        patient_id="H06",
        age=60, sex="M",
        chief_complaint="Chest pain, hoarseness, dysphagia — smoker",
        symptoms=[
            "sharp chest pain", "shortness of breath", "chest tightness",
            "fatigue", "jaundice", "dizziness", "sharp abdominal pain"
        ],
        vitals=Vitals(
            heart_rate=82, blood_pressure_sys=162, blood_pressure_dia=98,
            respiratory_rate=20, oxygen_saturation=95.0, temperature=36.9, gcs=15
        ),
        history="40-pack-year smoker. BP differential between arms noted by triage nurse.",
        true_disease="thoracic aortic aneurysm", true_esi=1,
    ),
    Patient(
        patient_id="H07",
        age=58, sex="M",
        chief_complaint="Severe epigastric pain radiating to back, alcohol binge",
        symptoms=[
            "burning abdominal pain", "upper abdominal pain", "back pain",
            "nausea", "vomiting", "abusing alcohol", "diarrhea",
            "side pain", "sharp abdominal pain", "lower body pain", "hemoptysis"
        ],
        vitals=Vitals(
            heart_rate=106, blood_pressure_sys=104, blood_pressure_dia=66,
            respiratory_rate=20, oxygen_saturation=96.0, temperature=38.2, gcs=14
        ),
        history="Chronic pancreatitis. Binge drinking last 3 days.",
        true_disease="acute pancreatitis", true_esi=2,
    ),
    Patient(
        patient_id="H08",
        age=76, sex="F",
        chief_complaint="Dizziness, weakness, dark stools x 3 days",
        symptoms=[
            "dizziness", "weakness", "melena", "changes in stool appearance",
            "vomiting blood", "nosebleed", "shortness of breath",
            "fatigue", "pallor"
        ],
        vitals=Vitals(
            heart_rate=100, blood_pressure_sys=102, blood_pressure_dia=64,
            respiratory_rate=18, oxygen_saturation=95.0, temperature=36.8, gcs=15
        ),
        history="On aspirin and warfarin for AF. INR last checked 2 weeks ago.",
        true_disease="anemia", true_esi=2,
    ),
    Patient(
        patient_id="H09",
        age=31, sex="M",
        chief_complaint="Right calf pain and swelling after long immobility",
        symptoms=[
            "leg pain", "leg swelling", "side pain", "shortness of breath",
            "sharp chest pain", "sweating", "cough", "weakness",
            "hemoptysis", "difficulty breathing"
        ],
        vitals=Vitals(
            heart_rate=98, blood_pressure_sys=118, blood_pressure_dia=76,
            respiratory_rate=20, oxygen_saturation=96.0, temperature=37.0, gcs=15
        ),
        history="Wheelchair user. Had minor leg surgery 2 weeks ago. Wells score = 4.",
        true_disease="pulmonary embolism", true_esi=2,
    ),
    Patient(
        patient_id="H10",
        age=82, sex="F",
        chief_complaint="Productive cough, low-grade fever, confusion",
        symptoms=[
            "cough", "fever", "difficulty breathing", "shortness of breath",
            "weakness", "chills", "vomiting", "wheezing",
            "nasal congestion", "sore throat"
        ],
        vitals=Vitals(
            heart_rate=90, blood_pressure_sys=130, blood_pressure_dia=80,
            respiratory_rate=24, oxygen_saturation=91.0, temperature=38.4, gcs=13
        ),
        history="CURB-65 score = 4. Lives alone. Nursing home referral pending.",
        true_disease="pneumonia", true_esi=2,
    ),
    Patient(
        patient_id="H11",
        age=39, sex="F",
        chief_complaint="Wheezing, tightness, cough — ran out of inhaler",
        symptoms=[
            "wheezing", "difficulty breathing", "cough", "chest tightness",
            "shortness of breath", "coughing up sputum", "nasal congestion",
            "allergic reaction", "sharp chest pain", "coryza"
        ],
        vitals=Vitals(
            heart_rate=110, blood_pressure_sys=132, blood_pressure_dia=86,
            respiratory_rate=30, oxygen_saturation=91.0, temperature=37.2, gcs=15
        ),
        history="Severe asthmatic. Last hospitalization 6 months ago.",
        true_disease="asthma", true_esi=3,
    ),
    Patient(
        patient_id="H12",
        age=50, sex="M",
        chief_complaint="Dysuria, flank pain, fever — recurrent kidney infections",
        symptoms=[
            "painful urination", "side pain", "fever", "back pain",
            "frequent urination", "blood in urine", "suprapubic pain",
            "nausea", "vomiting", "lower abdominal pain"
        ],
        vitals=Vitals(
            heart_rate=96, blood_pressure_sys=128, blood_pressure_dia=80,
            respiratory_rate=18, oxygen_saturation=98.0, temperature=38.6, gcs=15
        ),
        history="Recurrent UTIs. Immunocompromised (renal transplant).",
        true_disease="urinary tract infection", true_esi=3,
    ),
    Patient(
        patient_id="H13",
        age=25, sex="F",
        chief_complaint="Severe one-sided headache, visual aura, vomiting",
        symptoms=[
            "headache", "nausea", "vomiting", "blindness", "dizziness",
            "diminished vision", "spots or clouds in vision",
            "disturbance of memory", "symptoms of the face"
        ],
        vitals=Vitals(
            heart_rate=78, blood_pressure_sys=120, blood_pressure_dia=78,
            respiratory_rate=16, oxygen_saturation=99.0, temperature=36.8, gcs=15
        ),
        history="Classical migraines since teen years. Sumatriptan not available.",
        true_disease="migraine", true_esi=3,
    ),
    Patient(
        patient_id="H14",
        age=19, sex="M",
        chief_complaint="Nausea, vomiting, diarrhea — attended a festival",
        symptoms=[
            "nausea", "vomiting", "diarrhea", "decreased appetite",
            "sharp abdominal pain", "fever", "chills", "flu-like syndrome",
            "blood in stool", "fluid retention"
        ],
        vitals=Vitals(
            heart_rate=82, blood_pressure_sys=114, blood_pressure_dia=72,
            respiratory_rate=16, oxygen_saturation=99.0, temperature=37.9, gcs=15
        ),
        history="Otherwise healthy. Multiple others ill from same food vendor.",
        true_disease="infectious gastroenteritis", true_esi=4,
    ),
    Patient(
        patient_id="H15",
        age=24, sex="F",
        chief_complaint="Palpitations, hyperventilating, feels like dying",
        symptoms=[
            "palpitations", "breathing fast", "anxiety and nervousness",
            "chest tightness", "dizziness", "irregular heartbeat",
            "depressive or psychotic symptoms", "insomnia", "depression"
        ],
        vitals=Vitals(
            heart_rate=102, blood_pressure_sys=126, blood_pressure_dia=80,
            respiratory_rate=24, oxygen_saturation=98.0, temperature=36.7, gcs=15
        ),
        history="Known panic disorder. Stopped SSRI 2 weeks ago. No cardiac history.",
        true_disease="panic disorder", true_esi=5,
    ),
]

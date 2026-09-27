# ==========================================
# HEALTHCARE SYMPTOM PREDICTION CHATBOT
# ==========================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())


# ==========================================
# STEP 2: FIND TARGET COLUMN
# ==========================================

# Usually the target column is called "Disease"

target_column = "Disease"

# Remove unnecessary spaces from column names

data.columns = data.columns.str.strip()


# ==========================================
# STEP 3: GET SYMPTOM COLUMNS
# ==========================================

symptom_columns = []

for column in data.columns:

    if column != target_column:
        symptom_columns.append(column)


print("\nNumber of symptom columns:",
      len(symptom_columns))


# ==========================================
# STEP 4: CLEAN SYMPTOMS
# ==========================================

for column in symptom_columns:

    data[column] = data[column].fillna("")

    data[column] = data[column].astype(str)

    data[column] = data[column].str.strip()


# ==========================================
# STEP 5: CREATE SYMPTOM LIST
# ==========================================

X = []

for index, row in data.iterrows():

    symptoms = []

    for column in symptom_columns:

        symptom = row[column]

        if symptom != "":
            symptoms.append(symptom)

    X.append(symptoms)


y = data[target_column]


# ==========================================
# STEP 6: CONVERT SYMPTOMS TO NUMBERS
# ==========================================

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(X)


print("\nNumber of features:",
      len(mlb.classes_))


# ==========================================
# STEP 7: SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# STEP 8: TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# STEP 9: TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print("Accuracy:",
      round(accuracy * 100, 2),
      "%")


# ==========================================
# STEP 10: CHATBOT
# ==========================================

print("\n==========================================")
print("      HEALTHCARE SYMPTOM CHATBOT")
print("==========================================")

print("\nAvailable symptoms include:")

for symptom in mlb.classes_[:30]:

    print("-", symptom)


print("\nEnter your symptoms separated by commas.")

print("Example:")
print("itching, skin rash, fatigue")

print("\nType 'exit' to close.")


# ==========================================
# STEP 11: USER INPUT
# ==========================================

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":

        print("\nBot: Take care!")
        break


    # Convert input into list

    user_symptoms = user_input.lower().split(",")

    user_symptoms = [
        symptom.strip()
        for symptom in user_symptoms
    ]


    # ======================================
    # STEP 12: CREATE INPUT VECTOR
    # ======================================

    input_vector = []

    for symptom in mlb.classes_:

        if symptom.lower() in user_symptoms:

            input_vector.append(1)

        else:

            input_vector.append(0)


    # Convert to DataFrame/list format

    input_vector = [input_vector]


    # ======================================
    # STEP 13: PREDICT
    # ======================================

    prediction = model.predict(input_vector)


    print("\nBot: Based on the symptoms provided,")
    print("the model predicts:")

    print("\nPossible condition:",
          prediction[0])


    print("\nPlease note:")
    print("This is an educational machine-learning")
    print("project and is NOT a medical diagnosis.")
    print("Consult a qualified healthcare professional")
    print("for medical advice.")
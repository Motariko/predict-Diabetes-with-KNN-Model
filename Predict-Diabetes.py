import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("diabetes_del24.csv")

preg = int(input("Enter Pregnancies: "))
glu  = float(input("Enter Glucose: "))
bp   = float(input("Enter Blood Pressure: "))
skin = float(input("Enter Skin Thickness: "))
ins  = float(input("Enter Insulin: "))
bmi  = float(input("Enter BMI: "))
dpf  = float(input("Enter Diabetes Pedigree: "))
age  = int(input("Enter Age: "))

patient_data = [[preg, glu, bp, skin, ins, bmi, dpf, age]]

X = df.drop("Outcome", axis=1)
Y = df["Outcome"]
patient_df = pd.DataFrame(patient_data, columns=X.columns) 


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
patient_data_scaled = scaler.transform(patient_df)

K = 17
print(f"K ={K}")
knn = KNeighborsClassifier(n_neighbors=K, metric='euclidean')
knn.fit(X_scaled,Y)
print("Laening sacess")

y_pred = knn.predict(patient_data_scaled)

if y_pred[0] == 1:
    print("คนไข้อาจเป็นเบาหวาน")
else:
    print("คนไข้อาจไม่เป็นโรคเบาหวาน")#hello


                




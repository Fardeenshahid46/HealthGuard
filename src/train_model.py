import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

df=pd.read_csv("data/cleaned_diabetes.csv")

X=df.drop(columns=['Outcome'])
y=df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,)

#models
rf=RandomForestClassifier(n_estimators=200, random_state=42)
gb=GradientBoostingClassifier(random_state=42)

#model combination
ensemble=VotingClassifier(estimators=[('rf',rf),('gb',gb)],voting='soft')

#train the ensemble model
ensemble.fit(X_train, y_train)
y_pred=ensemble.predict(X_test)

#evaluation
accuracy=accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

#Save model
os.makedirs('src',exist_ok=True)
joblib.dump(ensemble,'src/model.pkl')
print("\n✅ Model saved to: src/model.pkl")

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "hotel_room_prices.csv")

X = df.drop(columns=["hotel_id", "room_price_per_night"])
y = df["room_price_per_night"]

categorical = ["city_type", "room_type", "season"]
numeric = [c for c in X.columns if c not in categorical]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, random_state=42, max_depth=12
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Hotel Room Price Prediction")
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.2f}")
print(f"R2 Score: {r2_score(y_test, pred):.4f}")

joblib.dump(model, BASE / "hotel_room_price_model.pkl")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.7)
plt.xlabel("Actual Room Price")
plt.ylabel("Predicted Room Price")
plt.title("Actual vs Predicted Hotel Room Price")
plt.tight_layout()
plt.savefig(BASE / "actual_vs_predicted.png")
print("Model and chart saved.")

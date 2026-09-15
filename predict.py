
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "hotel_room_price_model.pkl")

print("Enter hotel room details:")
city = input("City type (Budget City/Business City/Tourist City): ")
room_type = input("Room type (Standard/Deluxe/Suite): ")
season = input("Season (Low/Regular/Peak): ")
guests = int(input("Number of guests: "))
nights = int(input("Stay nights: "))
rating = float(input("Hotel rating (2.5-5.0): "))
distance = float(input("Distance from center (km): "))

sample = pd.DataFrame([{
    "city_type": city,
    "room_type": room_type,
    "season": season,
    "number_of_guests": guests,
    "stay_nights": nights,
    "hotel_rating": rating,
    "distance_from_center_km": distance
}])

prediction = model.predict(sample)[0]
print(f"\nPredicted Room Price Per Night: {prediction:.2f}")

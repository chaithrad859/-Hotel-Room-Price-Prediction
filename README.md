# Hotel Room Price Prediction

## Description
This machine learning project predicts the price of a hotel room per night based on hotel and booking details.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- OneHotEncoder
- Joblib
- Matplotlib

## Input Features
- City type
- Room type
- Season
- Number of guests
- Stay nights
- Hotel rating
- Distance from city center

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts the room price for a new booking
- `data/hotel_room_prices.csv` - synthetic dataset
- `hotel_room_price_model.pkl` - trained model
- `actual_vs_predicted.png` - evaluation chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

## Note
The dataset is synthetic and intended for educational demonstration purposes.

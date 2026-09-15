# -Hotel-Room-Price-Prediction
Hotel Room Price Prediction using Machine Learning and Scikit-learn — a Random Forest regression project with a Streamlit web app for estimating nightly hotel room prices.
#  Hotel Room Price Prediction using Scikit-learn

A machine learning project that predicts **hotel room prices per night** based on hotel, room, booking, seasonal, and location-related features.

The project uses a **Random Forest Regressor** with preprocessing through Scikit-learn pipelines and provides both a command-line prediction script and an interactive **Streamlit web application**.

---

##  Project Overview

Hotel room prices can vary depending on several factors such as:

* Destination/city type
* Room category
* Booking season
* Number of guests
* Length of stay
* Hotel rating
* Distance from the city center

This project applies machine learning to estimate the expected **room price per night** from these factors.

The project is designed as an educational machine learning application demonstrating:

* Data preprocessing
* Categorical feature encoding
* Regression modeling
* Model evaluation
* Model serialization
* Single-record prediction
* Batch hotel inventory valuation
* Interactive Streamlit deployment

---

##  Objectives

* Predict hotel room prices using machine learning.
* Handle categorical and numerical features efficiently.
* Build a reusable Scikit-learn preprocessing and modeling pipeline.
* Evaluate regression performance using standard metrics.
* Provide an easy-to-use web interface for price prediction.
* Support batch prediction using CSV hotel inventory data.

---

##  Machine Learning Model

The project uses:

**Random Forest Regressor**

Model configuration:

* `n_estimators = 200`
* `max_depth = 12`
* `random_state = 42`

Categorical features are processed using:

**OneHotEncoder**

The preprocessing and model are combined into a Scikit-learn **Pipeline**, making the complete prediction workflow reusable.

---

##  Dataset

The project includes a synthetic dataset containing **600 hotel listings**.

### Features

| Feature                   | Description                |
| ------------------------- | -------------------------- |
| `hotel_id`                | Unique hotel identifier    |
| `city_type`               | Type of destination market |
| `room_type`               | Standard, Deluxe, or Suite |
| `season`                  | Low, Regular, or Peak      |
| `number_of_guests`        | Number of guests           |
| `stay_nights`             | Number of nights           |
| `hotel_rating`            | Hotel rating               |
| `distance_from_center_km` | Distance from city center  |
| `room_price_per_night`    | Target room price          |

### Target Variable

`room_price_per_night`

The dataset is **synthetic** and is intended for educational and demonstration purposes.

---

##  Model Performance

The model was evaluated on a **20% test set**.

| Metric   |      Score |
| -------- | ---------: |
| MAE      |  **10.57** |
| RMSE     |  **13.22** |
| R² Score | **0.9613** |

### Metric Explanation

**MAE (Mean Absolute Error)**
Measures the average absolute difference between actual and predicted room prices.

**RMSE (Root Mean Squared Error)**
Measures prediction error while giving greater weight to larger errors.

**R² Score**
Indicates how well the model explains the variation in hotel room prices. A score of **0.9613** indicates strong performance on this dataset.

---

##  Streamlit Web Application

The project includes an interactive Streamlit application with three main sections:

###  Room Rate Calculator

Users can enter:

* Destination market type
* Hotel rating
* Distance from city center
* Room category
* Booking season
* Number of guests
* Length of stay

The application calculates:

* Estimated nightly room price
* Estimated total stay cost
* Pricing factors and insights
* Recommended competitive rate range

###  Batch Inventory Pricing

Users can upload a hotel inventory CSV and generate predictions for multiple listings.

The application calculates:

* Predicted price per night
* Estimated total booking cost
* Average nightly rate
* Batch pricing results

###  Model Performance

The application also provides model-performance and regression-analysis information.

---

##  Project Structure

```text
Hotel_Room_Price_Prediction_Sklearn/
│
├── app.py
├── train_model.py
├── predict.py
├── hotel_room_price_model.pkl
├── actual_vs_predicted.png
├── requirements.txt
│
└── data/
    └── hotel_room_prices.csv
```

---

##  Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Random Forest**
* **OneHotEncoder**
* **Joblib**
* **Matplotlib**
* **Streamlit**

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Hotel-Room-Price-Prediction-Sklearn.git
```

Navigate into the project:

```bash
cd Hotel-Room-Price-Prediction-Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

##  Train the Model

To train the model from the dataset:

```bash
python train_model.py
```

This will:

1. Load the hotel dataset.
2. Separate features and target.
3. Encode categorical variables.
4. Split the data into training and testing sets.
5. Train the Random Forest Regressor.
6. Evaluate the model.
7. Save the trained model as `hotel_room_price_model.pkl`.
8. Generate the actual-vs-predicted visualization.

---

##  Make a Prediction

Run:

```bash
python predict.py
```

The script asks for hotel and booking information such as:

```text
City type
Room type
Season
Number of guests
Stay nights
Hotel rating
Distance from city center
```

It then returns the predicted room price per night.

---

##  Run the Streamlit Application

Start the web application with:

```bash
streamlit run app.py
```

The application will open in your browser and provide an interactive hotel pricing interface.

---

##  Prediction Workflow

```text
Hotel & Booking Data
        ↓
Data Preprocessing
        ↓
Categorical Encoding
        ↓
Train/Test Split
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Saved ML Model
        ↓
Price Prediction
        ↓
Streamlit Web Application
```

---

##  Example Use Case

A hotel manager can provide information such as:

```text
City Type       : Tourist City
Room Type       : Suite
Season          : Peak
Guests          : 2
Stay Nights     : 4
Hotel Rating    : 4.8
Distance        : 1.2 km
```

The trained model can then estimate the expected nightly room rate and calculate the approximate total cost for the stay.

---

##  Key Features

*  Random Forest regression
*  Numerical and categorical feature handling
*  One-hot encoding
*  Scikit-learn Pipeline
*  Model evaluation with MAE, RMSE, and R²
*  Saved `.pkl` machine learning model
*  Command-line prediction
*  Interactive Streamlit application
*  Batch CSV price prediction
*  Actual vs. predicted visualization
*  Synthetic dataset for educational use

---

##  Disclaimer

This project uses a **synthetic hotel pricing dataset**. The predictions are intended for educational and demonstration purposes and should not be treated as real-world hotel pricing recommendations.

---

##  Author

**Your Name**

If you found this project useful, consider giving the repository a .

---

## 📜 License

This project is available for educational and learning purposes.

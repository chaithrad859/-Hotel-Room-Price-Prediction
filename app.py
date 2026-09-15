import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Hotel Room Price Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Hospitality Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border: 1px solid rgba(165, 180, 252, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(165, 180, 252, 0.15);
        color: #C7D2FE;
        border: 1px solid rgba(165, 180, 252, 0.35);
        margin-bottom: 10px;
    }

    .price-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(67, 56, 202, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 16px;
        padding: 26px;
        text-align: center;
    }

    .price-hero {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #818CF8;
        margin: 6px 0;
    }

    .breakdown-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find assets
def get_asset_path(filename):
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(filename),
        script_dir / filename,
        script_dir / "Hotel_Room_Price_Prediction_Sklearn" / filename,
        script_dir.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return filename

@st.cache_resource
def load_model():
    model_path = get_asset_path("hotel_room_price_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Revenue Management & Hospitality AI</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🏨 Hotel Room Price Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast dynamic hotel room rates per night and total booking costs based on destination city, room category, seasonality, rating, and location proximity.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🛎️ Room Rate Calculator", "📁 Batch Inventory Pricing (CSV)", "📊 Model Performance & Regression Analysis"])

# --- TAB 1: Room Rate Calculator ---
with tabs[0]:
    st.subheader("Hotel Property & Reservation Details")

    # Quick Presets
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        preset_luxury = st.button("🏖️ Peak Tourist Luxury Suite", width="stretch")
    with p_cols[1]:
        preset_biz = st.button("💼 Business Deluxe Downtown", width="stretch")
    with p_cols[2]:
        preset_budget = st.button("🎒 Budget Standard Room", width="stretch")

    if preset_luxury:
        st.session_state["city"] = "Tourist City"
        st.session_state["room"] = "Suite"
        st.session_state["season"] = "Peak"
        st.session_state["guests"] = 2
        st.session_state["nights"] = 4
        st.session_state["rating"] = 4.8
        st.session_state["distance"] = 1.2
    elif preset_biz:
        st.session_state["city"] = "Business City"
        st.session_state["room"] = "Deluxe"
        st.session_state["season"] = "Regular"
        st.session_state["guests"] = 1
        st.session_state["nights"] = 3
        st.session_state["rating"] = 4.2
        st.session_state["distance"] = 2.5
    elif preset_budget:
        st.session_state["city"] = "Budget City"
        st.session_state["room"] = "Standard"
        st.session_state["season"] = "Low"
        st.session_state["guests"] = 2
        st.session_state["nights"] = 2
        st.session_state["rating"] = 3.2
        st.session_state["distance"] = 8.5

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### 📍 Destination & Hotel Quality")
        city_type = st.selectbox(
            "Destination Market Type",
            ["Tourist City", "Business City", "Budget City"],
            index=["Tourist City", "Business City", "Budget City"].index(st.session_state.get("city", "Tourist City")),
            key="input_city"
        )
        hotel_rating = st.slider(
            "Hotel Star Rating (2.5 - 5.0 ⭐)", 2.5, 5.0,
            value=float(st.session_state.get("rating", 4.2)), step=0.1,
            key="input_rating"
        )
        distance = st.slider(
            "Distance From City Center (km)", 0.5, 25.0,
            value=float(st.session_state.get("distance", 3.0)), step=0.5,
            key="input_distance", help="Proximity to primary urban/tourist center."
        )

    with c_right:
        st.markdown("#### 🛏️ Room Type & Booking Configuration")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            room_type = st.selectbox(
                "Room Category",
                ["Standard", "Deluxe", "Suite"],
                index=["Standard", "Deluxe", "Suite"].index(st.session_state.get("room", "Deluxe")),
                key="input_room"
            )
        with col_s2:
            season = st.selectbox(
                "Booking Season",
                ["Low", "Regular", "Peak"],
                index=["Low", "Regular", "Peak"].index(st.session_state.get("season", "Regular")),
                key="input_season"
            )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            guests = st.slider(
                "Number of Guests", 1, 6,
                value=int(st.session_state.get("guests", 2)), step=1,
                key="input_guests"
            )
        with col_b2:
            stay_nights = st.slider(
                "Length of Stay (Nights)", 1, 30,
                value=int(st.session_state.get("nights", 3)), step=1,
                key="input_nights"
            )

    st.markdown("---")
    predict_btn = st.button("🚀 Calculate Estimated Room Price", type="primary", width="stretch")

    booking_df = pd.DataFrame([{
        "city_type": city_type,
        "room_type": room_type,
        "season": season,
        "number_of_guests": guests,
        "stay_nights": stay_nights,
        "hotel_rating": hotel_rating,
        "distance_from_center_km": distance
    }])

    pred_price = max(0.0, float(model.predict(booking_df)[0]))
    total_cost = pred_price * stay_nights

    st.markdown("### 📋 Pricing Outcome & Stay Estimate")
    r1, r2 = st.columns([1.3, 1.7], gap="medium")

    with r1:
        st.markdown(f"""
        <div class="price-card">
            <span style="font-size: 2.8rem;">🛎️</span>
            <div style="color: #A5B4FC; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">
                Estimated Nightly Rate
            </div>
            <div class="price-hero">
                ${pred_price:.2f} <span style="font-size: 1.2rem; color: #94A3B8;">/ night</span>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 10px; margin-top: 10px;">
                <span style="color: #94A3B8; font-size: 0.95rem;">Total Stay ({stay_nights} nights):</span>
                <strong style="color: #34D399; font-size: 1.25rem;"> ${total_cost:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown("#### 💡 Value Drivers & Market Factors")
        drivers = []
        if season == "Peak":
            drivers.append(("Peak Holiday Surcharge", "High traveler influx triggers peak market pricing.", "warn"))
        elif season == "Low":
            drivers.append(("Off-Peak Discounting", "Lower baseline market rates available during off-season.", "info"))

        if room_type == "Suite":
            drivers.append(("Luxury Suite Premium", "Premium square footage, view, and concierge access applied.", "warn"))
        elif room_type == "Deluxe":
            drivers.append(("Deluxe Upgrade", "Higher-tier amenities with moderate upgrade rate.", "info"))

        if distance < 2.0:
            drivers.append(("Prime City Center Proximity", f"Only {distance} km from center; prime walkable location value.", "info"))
        elif distance > 10.0:
            drivers.append(("Suburban / Outer Location", f"{distance} km from downtown enables cost-effective room rates.", "info"))

        if hotel_rating >= 4.5:
            drivers.append(("Five-Star Service Standard", f"{hotel_rating}⭐ rating commands elite hospitality pricing.", "warn"))

        if drivers:
            for title, desc, tone in drivers:
                if tone == "warn":
                    st.warning(f"**{title}**: {desc}")
                else:
                    st.info(f"**{title}**: {desc}")
        else:
            st.info("Pricing corresponds with standard baseline seasonal averages.")

        st.markdown(f"""
        <div class="breakdown-box">
            <strong style="color: #A5B4FC;">Revenue Management Insight:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                Target Average Daily Rate (ADR) benchmark: <strong>${pred_price:.2f}</strong>. Recommended competitive rate band: <strong>${pred_price * 0.92:.2f} - ${pred_price * 1.08:.2f}</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Features"):
        st.dataframe(booking_df, width="stretch")

# --- TAB 2: Batch Inventory Pricing ---
with tabs[1]:
    st.subheader("Batch Property Inventory Pricing")
    st.write("Upload a hotel inventory CSV or evaluate against the 600-listing hotel room dataset.")

    csv_file = st.file_uploader("Upload Hotel Inventory CSV", type=["csv"], key="hotel_csv")
    df_hotels = None

    if csv_file is not None:
        df_hotels = pd.read_csv(csv_file)
        st.info(f"Loaded {len(df_hotels)} hotel listings from file.")
    else:
        sample_path = get_asset_path("data/hotel_room_prices.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Load baseline hotel dataset (`data/hotel_room_prices.csv`)", value=True):
                df_hotels = pd.read_csv(sample_path)
                st.info(f"Loaded {len(df_hotels)} records from baseline dataset.")

    if df_hotels is not None:
        req_cols = ["city_type", "room_type", "season", "number_of_guests", "stay_nights", "hotel_rating", "distance_from_center_km"]
        missing = [c for c in req_cols if c not in df_hotels.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Batch Valuation", type="primary"):
                with st.spinner("Valuating hotel rooms..."):
                    preds = model.predict(df_hotels[req_cols])
                    preds = np.maximum(0.0, preds)

                    res_df = df_hotels.copy()
                    res_df["Predicted_Price_Per_Night"] = np.round(preds, 2)
                    res_df["Total_Estimated_Cost"] = np.round(preds * res_df["stay_nights"], 2)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Listings", len(res_df))
                    m2.metric("Average Nightly Rate", f"${np.mean(preds):.2f}")
                    m3.metric("Highest Room Rate", f"${np.max(preds):.2f}")
                    m4.metric("Lowest Room Rate", f"${np.min(preds):.2f}")

                    f1, f2 = st.columns(2)
                    with f1:
                        city_f = st.selectbox("Filter City Type:", ["All"] + list(df_hotels["city_type"].unique()))
                    with f2:
                        room_f = st.selectbox("Filter Room Type:", ["All"] + list(df_hotels["room_type"].unique()))

                    filtered = res_df
                    if city_f != "All":
                        filtered = filtered[filtered["city_type"] == city_f]
                    if room_f != "All":
                        filtered = filtered[filtered["room_type"] == room_f]

                    st.dataframe(filtered, width="stretch")

                    csv_export = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Pricing Valuations as CSV",
                        data=csv_export,
                        file_name="hotel_room_pricing_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Accuracy Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🤖 Valuation Model Specifications
        - **Algorithm**: `RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)`
        - **Pipeline Preprocessing**:
            - `OneHotEncoder` on categorical factors (`city_type`, `room_type`, `season`)
            - Passthrough on numerical attributes (`number_of_guests`, `stay_nights`, `hotel_rating`, `distance_from_center_km`)
        - **Accuracy Benchmark**:
            - **R² Score**: **0.9610** (96.1% variance explained)
            - **MAE (Mean Absolute Error)**: ~$10.63
            - **RMSE**: ~$13.26
        """)

    with c2:
        img_path = get_asset_path("actual_vs_predicted.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Actual vs. Predicted Hotel Room Prices", width="stretch")
        else:
            st.info("Chart image not found.")

st.caption("Hotel Hospitality Dynamic Pricing Intelligence • Scikit-learn & Streamlit")

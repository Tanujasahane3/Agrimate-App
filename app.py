import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AgriMate - Crop Profit Estimator",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Background and font */
    .main {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
    }
    /* Headings */
    h1, h2, h3 {
        color: #2a9d8f;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 700;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21867a;
        cursor: pointer;
    }
    /* Footer style */
    footer {
        margin-top: 50px;
        font-style: italic;
        color: #555;
    }
    /* Tooltip for sidebar inputs */
    [data-tooltip] {
      position: relative;
      cursor: help;
    }
    [data-tooltip]:hover::after {
      content: attr(data-tooltip);
      position: absolute;
      left: 100%;
      margin-left: 10px;
      top: 50%;
      transform: translateY(-50%);
      background: #2a9d8f;
      color: white;
      padding: 5px 10px;
      border-radius: 6px;
      white-space: nowrap;
      z-index: 1000;
      font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING FUNCTION ---
@st.cache_data(show_spinner=True)
def load_data():
    crop_df = pd.read_csv("crop_data.csv")
    mandi_df = pd.read_csv("mandi_prices.csv")
    yield_mod = joblib.load("yield_model.pkl")
    seed_enc = joblib.load("seed_type_encoder.pkl")
    return crop_df, mandi_df, yield_mod, seed_enc

try:
    crop_data, mandi_data, yield_model, seed_type_encoder = load_data()
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Enter Crop Details")

crop = st.sidebar.selectbox(
    label="Select Crop",
    options=crop_data['Crop'].unique(),
    help="Choose the crop you want to estimate profit for."
)

area = st.sidebar.slider(
    label="Area (in acres)",
    min_value=0.1,
    max_value=50.0,
    value=1.0,
    step=0.1,
    help="Enter the area size of your crop."
)

seed_type = st.sidebar.selectbox(
    label="Seed Type",
    options=["Hybrid", "Organic", "Local"],
    help="Select seed type for your crop."
)

location = st.sidebar.text_input(
    label="Location (e.g., Nashik)",
    help="Enter your location for accurate market prices."
)

show_crop_data = st.sidebar.checkbox("Show Sample Crop Data")
show_mandi_data = st.sidebar.checkbox("Show Sample Market Price Data")

# --- HEADER & INTRO ---
st.title("🌾 AgriMate - Crop Budget & Profit Estimator")
st.markdown("""
Welcome to **AgriMate**, your trusted assistant in smart farming.  
Easily estimate your crop profitability based on your inputs and local market data.  
Make informed decisions and grow with confidence!  
""")

# --- DATA PREVIEWS ---
if show_crop_data:
    with st.expander("Sample Crop Data"):
        st.dataframe(crop_data.head())

if show_mandi_data:
    with st.expander("Sample Market Price Data"):
        st.dataframe(mandi_data.head())

# --- HELPER FUNCTION TO PREDICT ---
def predict_profit(crop, area, seed_type, location):
    # Filter crop info
    crop_info = crop_data[crop_data['Crop'] == crop].iloc[0]
    seed_cost = crop_info['Seed_Cost_per_Acre']
    fertilizer_cost = crop_info['Fertilizer_Cost']
    expected_yield_per_acre = crop_info['Expected_Yield_per_Acre']

    input_cost = (seed_cost + fertilizer_cost) * area

    # Encode seed type
    seed_encoded = seed_type_encoder.transform([seed_type])[0]
    X = np.array([[area, seed_encoded]])

    predicted_yield_per_acre = yield_model.predict(X)[0]
    predicted_yield = predicted_yield_per_acre * area

    # Market price based on location and crop
    mandi_price = mandi_data[
        (mandi_data['Crop'] == crop) &
        (mandi_data['Location'].str.lower() == location.strip().lower())
    ]['Market_Price']
    if not mandi_price.empty:
        price = mandi_price.values[0]
    else:
        price = mandi_data[mandi_data['Crop'] == crop]['Market_Price'].mean()

    est_income = predicted_yield * price
    profit = est_income - input_cost

    return {
        "input_cost": input_cost,
        "predicted_yield": predicted_yield,
        "market_price": price,
        "estimated_income": est_income,
        "profit": profit
    }

# --- PREDICTION & RESULTS ---
if st.button("Estimate Profit"):
    if location.strip() == "":
        st.error("Please enter a location for accurate market price.")
    else:
        try:
            results = predict_profit(crop, area, seed_type, location)
            st.success("Estimation Complete! Here's your report:")

            # Show metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Input Cost (Rs.)", f"{results['input_cost']:,.2f}")
                st.metric("Predicted Yield (quintals)", f"{results['predicted_yield']:.2f}")
                st.metric("Market Price (Rs./quintal)", f"{results['market_price']:,.2f}")
            with col2:
                st.metric("Estimated Income (Rs.)", f"{results['estimated_income']:,.2f}")
                st.metric("Estimated Profit (Rs.)", f"{results['profit']:,.2f}")

            # Bar Chart: Input Cost vs Income
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                ['Input Cost', 'Estimated Income'],
                [results['input_cost'], results['estimated_income']],
                color=['#e76f51', '#2a9d8f']
            )
            ax.set_title("Input Cost vs Estimated Income")
            ax.set_ylabel("Amount (Rs.)")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            st.pyplot(fig)

            # Pie chart: Cost components
            pie_labels = ['Seed Cost', 'Fertilizer Cost']
            seed_cost_total = crop_data[crop_data['Crop'] == crop]['Seed_Cost_per_Acre'].values[0] * area
            fertilizer_cost_total = crop_data[crop_data['Crop'] == crop]['Fertilizer_Cost'].values[0] * area
            pie_sizes = [seed_cost_total, fertilizer_cost_total]

            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=['#264653', '#2a9d8f'])
            ax2.axis('equal')
            ax2.set_title("Input Cost Breakdown")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error in calculation: {e}")

# --- EXTRA INFO TABS ---
with st.expander("More Info and Tips 📚"):
    tabs = st.tabs(["Crop Info", "FAQs", "Contact"])
    
    with tabs[0]:
        st.subheader(f"{crop} Info & Growing Tips")
        # Dummy tips; you can replace with real tips from a dataset or API
        tips = {
            "Wheat": "Best grown in well-drained loamy soil, needs moderate water, harvest when grains harden.",
            "Rice": "Requires plenty of water, best in clayey soil, maintain standing water for growth.",
            "Maize": "Prefers fertile, well-drained soil; ensure adequate nitrogen fertilization.",
            "Sugarcane": "Needs rich soil with good moisture, plant in furrows, control weeds regularly."
        }
        st.write(tips.get(crop, "Detailed growing tips coming soon!"))

    with tabs[1]:
        st.subheader("Frequently Asked Questions")
        faqs = {
            "How accurate is the profit estimate?": "Estimates are based on historical data and models; real-world factors can vary results.",
            "Can I use this app for any crop?": "Currently supports selected crops in the dataset; more crops will be added soon.",
            "What does 'Seed Type' mean?": "Seed Type indicates whether the seed is hybrid, organic, or local variety.",
            "How is market price determined?": "Based on recent mandi data from your location; if unavailable, average price is used."
        }
        for q, a in faqs.items():
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

    with tabs[2]:
      st.subheader("Contact / Feedback")
    st.markdown("""
    If you have suggestions, feedback, or want to collaborate, reach out!  
    📧 Email: <a href='mailto:sahanetanuja5@gmail.com'>sahanetanuja5@gmail.com</a>
    """, unsafe_allow_html=True)   

# --- ABOUT SECTION ---
with st.expander("About This App 🧐"):
    st.write("""
    AgriMate is designed to empower farmers and agricultural stakeholders with smart insights into crop economics.  
    By leveraging machine learning and real-time market data, AgriMate helps optimize decisions to maximize profitability and sustainability.  
    Developed with passion and care to support the backbone of our nation — our farmers.
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<center><h4 style='color:#2a9d8f;'>“Empowering farmers, enriching lives — one crop at a time.”</h4></center>",
    unsafe_allow_html=True,
)
st.markdown("<center><i>Developed with ❤️ by Tanuja Sahane</i></center>", unsafe_allow_html=True)

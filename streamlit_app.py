import streamlit as st
from rag_search import get_stock_predictions

# Title of the app
st.title("The Wolf of r/wallstreetbets")

# Add text
st.write("Your own personalized stock predictor")
st.write("")
st.write("")

st.sidebar.header("Portfolio")

# Initialize portfolio in session state if not already created
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

# Add an input field
st.write("Let us know what you have in your portfolio: ")
col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA):", max_chars=5).upper()

with col2:
    shares = st.number_input("Number of Shares:", min_value=1, step=1)

button_col1, button_col2 = st.columns([1, 4])

with button_col1:
    add_clicked = st.button("Add to Portfolio")

with button_col2:
    generate_clicked = st.button("Generate Predictions")

# Handle button actions
if add_clicked:
    if ticker:
        st.session_state.portfolio[ticker] = shares

if generate_clicked:
    portfolio = []
    for tick, num_shares in st.session_state.portfolio.items():
        plural = "s" if num_shares > 1 else ""
        portfolio.append(f"{tick}: {num_shares} share{plural}")

    portfolio = ", ".join(portfolio)
    response = get_stock_predictions(portfolio)

    st.subheader("Investment Reccomendations:")
    st.write(response)

if st.session_state.portfolio:
    for tick, num_shares in st.session_state.portfolio.items():
        st.sidebar.write(f"Stock: **{tick}** Shares: **{num_shares}**")
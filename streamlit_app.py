import streamlit as st
from rag_search import get_stock_predictions

# Title of the app
st.title("The Wolf of r/wallstreetbets")

# Add text
st.write("Your own personalized stock predictor")
st.write("")
st.write("")

# Add custom CSS to style only the remove buttons
st.markdown("""
<style>
section[data-testid="stSidebar"] button {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #999 !important;
    font-size: 14px !important;
    padding: 0px !important;
    width: auto !important;
    height: auto !important;
    min-height: 0px !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: transparent !important;
    color: #ff0000 !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Portfolio")

# Initialize portfolio in session state if not already created
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

# Add a function to remove a stock from the portfolio
def remove_stock(ticker):
    if ticker in st.session_state.portfolio:
        del st.session_state.portfolio[ticker]

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

    st.subheader("Investment Recommendations:")
    st.write(response)

if st.session_state.portfolio:
    for tick, num_shares in list(st.session_state.portfolio.items()):
        # Use very small columns to position items side by side
        cols = st.sidebar.columns([15, 1])
        
        # Stock info in first column
        with cols[0]:
            st.write(f"**{tick}**: {num_shares} share{'s' if num_shares > 1 else ''}")
        
        # Remove button in second column (very small)
        with cols[1]:
            if st.button("×", key=f"remove_{tick}"):
                remove_stock(tick)
                st.rerun()
    
    print(st.session_state.portfolio)
else:
    st.sidebar.write("Your portfolio is empty.")
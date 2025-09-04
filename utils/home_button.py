import streamlit as st

def home_button(label="🏠 Return to Home"):
    if st.button(label):
        st.session_state.current_view = "home"
        st.experimental_rerun()  # instantly go to home

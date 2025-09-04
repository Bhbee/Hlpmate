import streamlit as st

def reset_app_session():
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    st.cache_data.clear()  # Optional: clears any cache you might use
    st.rerun()

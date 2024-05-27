import streamlit as st

def show_welcome_page():
    st.title("Welcome to the Cornish Dialect Fine-Tuning App")
    
    st.markdown("""
        ### Instructions
        1. Enter your Hugging Face token (it will be kept secret).
        2. Provide a model name for saving to the Hugging Face Hub.
        3. Click "Next" to proceed to data selection.
    """)
    
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    model_name = st.text_input("Enter the model name (for Hugging Face Hub)")

    if st.button("Next"):
        if hf_token and model_name:
            st.session_state.hf_token = hf_token
            st.session_state.model_name = model_name
            st.session_state.page = "data"
            st.experimental_rerun()
        else:
            st.error("Please enter both your Hugging Face token and model name.")

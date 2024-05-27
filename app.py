import streamlit as st
from welcome_page import show_welcome_page
from data_page import show_data_page
from fine_tune_page import show_fine_tune_page
from train_from_scratch_page import show_train_from_scratch_page
from train_with_lora_adapters_page import show_train_with_lora_adapters_page

def main():
    st.set_page_config(page_title="Dialect Adaptation for Large Language Models", layout="centered")

    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        show_welcome_page()
    elif st.session_state.page == "data":
        show_data_page()
    elif st.session_state.page == "fine_tune":
        show_fine_tune_page()
    elif st.session_state.page == "train_from_scratch":
        show_train_from_scratch_page()
    elif st.session_state.page == "train_with_lora_adapters":
        show_train_with_lora_adapters_page()

if __name__ == "__main__":
    main()

import streamlit as st

def show_fine_tune_page():
    st.title("Fine-Tune Options")

    option = st.selectbox("Select Fine-Tuning Method", [
        "Train from scratch",
        "Train with LoRA Adapters"
    ])

    if st.button("Next"):
        if option == "Train from scratch":
            st.session_state.page = "train_from_scratch"
        elif option == "Train with LoRA Adapters":
            st.session_state.page = "train_with_lora_adapters"
        st.experimental_rerun()

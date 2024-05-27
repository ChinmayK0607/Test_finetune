import streamlit as st
import subprocess
import sys

def show_train_from_scratch_page():
    st.title("Train from Scratch")

    num_steps = st.number_input("Enter the number of training steps", min_value=1, value=200)
    push_option = st.selectbox("Push options", [
        "Push entire model to hub",
        "Push LoRA adapters to hub"
    ])

    if st.button("Start Training"):
        hf_token = st.session_state.hf_token
        model_name = st.session_state.model_name
        csv_path = "data/processed/processed_" + st.session_state.uploaded_file_name
        
        # Determine the push options
        push_entire_model = "true" if push_option == "Push entire model to hub" else "false"
        push_lora_adapters = "true" if push_option == "Push LoRA adapters to hub" else "false"

        # Run the finetuning script
        command = [
            sys.executable, 'finetuning.py', hf_token, csv_path, model_name, 
            str(num_steps), push_entire_model, push_lora_adapters
        ]
        subprocess.Popen(command)
        
        st.success(f"Started training for {num_steps} steps with option to {push_option}.")

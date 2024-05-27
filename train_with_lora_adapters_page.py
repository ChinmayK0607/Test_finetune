import streamlit as st
import subprocess
import sys

def show_train_with_lora_adapters_page():
    st.title("Train with LoRA Adapters")

    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    repo_name = st.text_input("Enter the repo name for the model (for Hugging Face Hub)")
    num_steps = st.number_input("Enter the number of training steps", min_value=1, value=200)
    push_option = st.selectbox("Push options", [
        "Push entire model to hub",
        "Push LoRA adapters to hub"
    ])

    if st.button("Start Training"):
        if hf_token and repo_name:
            csv_path = "data/processed/processed_" + st.session_state.uploaded_file_name
            
            # Determine the push options
            push_entire_model = "true" if push_option == "Push entire model to hub" else "false"
            push_lora_adapters = "true" if push_option == "Push LoRA adapters to hub" else "false"

            # Run the lora_finetuning.py script
            command = [
                sys.executable, 'lora_finetuning.py', hf_token, csv_path, repo_name, 
                str(num_steps), push_entire_model, push_lora_adapters
            ]
            subprocess.Popen(command)
            
            st.success(f"Started training for {num_steps} steps with option to {push_option}.")
        else:
            st.error("Please provide the required Hugging Face token and repo name.")

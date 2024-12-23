import streamlit as st
import subprocess
import os

def run_deepfake_detection(input_path, output_path, info_path, device, checkpoint_path):
    # Prepare the command with the received arguments
    command = [
        "python", "main.py",
        "--input_path", input_path,
        "--output_path", output_path,
        "--info_path", info_path,
        "--device", device,
        "--checkpoint_path", checkpoint_path
    ]
    
    # Run the main.py script with the specified arguments
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Show the output or error messages from the subprocess
    return result.stdout, result.stderr

def main():
    st.title("Deepfake Detection with Streamlit")
    
    # Input fields in Streamlit for the user to provide input
    input_path = st.text_input("Input Path (Video file)", "data/shining.mp4")
    output_path = st.text_input("Output Path", "results")
    info_path = st.text_input("Info Path (JSON file)", "method_info.json")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    checkpoint_path = st.text_input("Checkpoint Path", "checkpoints/model.pth")
    
    # Button to start processing
    if st.button("Run Detection"):
        if input_path and output_path and info_path and checkpoint_path:
            st.write("Running deepfake detection...")
            stdout, stderr = run_deepfake_detection(input_path, output_path, info_path, device, checkpoint_path)
            
            if stderr:
                st.error(f"Error: {stderr}")
            else:
                st.success("Deepfake detection completed!")
                st.text_area("Output", stdout, height=300)
        else:
            st.error("Please fill in all the fields!")

if __name__ == "__main__":
    main()

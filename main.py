import streamlit as st
import os
import serial
from pain_recognition import process_video
from pose_estimation import run_pose_estimation
from plantar_pressure import run_plantar_pressure_analysis  # Import the plantar pressure function

def create_patient_folder(patient_name, patient_age):
    patient_folder = os.path.join("data", "patient_data", f"{patient_name}_{patient_age}")
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    return patient_folder

def main():
    st.title("OrthoSynergy")

    if 'create_new_patient' not in st.session_state:
        st.session_state.create_new_patient = False
        st.session_state.patient_name = ""
        st.session_state.patient_age = ""
        st.session_state.patient_folder = ""

    if st.session_state.create_new_patient:
        # Create tabs instead of radio buttons
        tab1, tab2, tab3, tab4 = st.tabs(["Pain Recognition", "Plantar Pressure", "Pose Estimation", "Generate Report"])

        with tab1:
            st.header("Pain Recognition")
            process_video(st.session_state.patient_folder)

        with tab2:
            st.header("Plantar Pressure")
            if st.button("Start Plantar Pressure Analysis"):
                try:
                    # Only open the serial port when this tab is selected
                    run_plantar_pressure_analysis()  # Call the plantar pressure analysis function
                except serial.SerialException as e:
                    st.error(f"Error opening serial port: {e}")

        with tab3:
            st.header("Pose Estimation")
            run_pose_estimation(st.session_state.patient_folder)  # Same structure as Pain Recognition

        with tab4:
            st.header("Generate Report")
            if st.button("Generate Report"):
                st.success("Report generation logic needs to be implemented.")
                
            if st.button("Reset for New Patient"):
                st.session_state.create_new_patient = False
                st.session_state.patient_name = ""
                st.session_state.patient_age = ""
                st.session_state.patient_folder = ""
                st.write("You can now create a new patient.")
                for key in list(st.session_state.keys()):
                    del st.session_state[key]

    else:
        st.sidebar.title("New Patient")
        st.session_state.patient_name = st.sidebar.text_input("Patient Name")
        st.session_state.patient_age = st.sidebar.text_input("Patient Age")

        if st.sidebar.button("Create New Patient"):
            if st.session_state.patient_name and st.session_state.patient_age:
                st.session_state.patient_folder = create_patient_folder(st.session_state.patient_name, st.session_state.patient_age)
                st.session_state.create_new_patient = True
                st.sidebar.empty()
            else:
                st.sidebar.warning("Please enter both patient name and age.")

if __name__ == "__main__":
    main()

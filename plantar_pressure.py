import streamlit as st
import serial
import numpy as np
import plotly.express as px
import time
import threading

def run_plantar_pressure_analysis():
    # Initialize serial connections for left and right foot
    ser_left = None
    ser_right = None
    try:
        ser_left = serial.Serial('COM4', 9600, timeout=1)  # Adjust 'COM4' to your left foot serial port
    except serial.SerialException as e:
        st.error("Left foot COM port not connected or unavailable.")
    
    try:
        ser_right = serial.Serial('COM6', 9600, timeout=1)  # Adjust 'COM6' to your right foot serial port
    except serial.SerialException as e:
        st.error("Right foot COM port not connected or unavailable.")

    # Initialize pressure matrices for left and right foot
    global pressure_matrix_left, pressure_matrix_right
    pressure_matrix_left = np.zeros((6, 3)) if ser_left else None
    pressure_matrix_right = np.zeros((6, 3)) if ser_right else None

    # Function to read pressure matrices from both Arduinos
    def read_pressure_matrices():
        global pressure_matrix_left, pressure_matrix_right  # Declare global variables
        while True:
            if ser_left:
                new_matrix_left = np.zeros((6, 3))
                # Read from the left foot Arduino
                for i in range(6):
                    try:
                        line_left = ser_left.readline().decode().strip()
                        if line_left:
                            pressure_row_left = list(map(int, line_left.split(',')))
                            if len(pressure_row_left) == 3:  # Expecting 3 readings for left foot
                                new_matrix_left[i] = pressure_row_left
                    except Exception as e:
                        print("Error reading from left foot Arduino:", e)
                pressure_matrix_left[:] = new_matrix_left
            
            if ser_right:
                new_matrix_right = np.zeros((6, 3))
                # Read from the right foot Arduino
                for i in range(6):
                    try:
                        line_right = ser_right.readline().decode().strip()
                        if line_right:
                            pressure_row_right = list(map(int, line_right.split(',')))
                            if len(pressure_row_right) == 3:  # Expecting 3 readings for right foot
                                new_matrix_right[i] = pressure_row_right
                    except Exception as e:
                        print("Error reading from right foot Arduino:", e)
                pressure_matrix_right[:] = new_matrix_right
            
            time.sleep(0.01)  # Update every 30 milliseconds

    # Start the serial reading in a separate thread if at least one device is connected
    if ser_left or ser_right:
        thread = threading.Thread(target=read_pressure_matrices, daemon=True)
        thread.start()

    # Create a layout with two columns for side-by-side heatmaps
    col1, col2 = st.columns(2)

    # Create placeholders for the heatmaps
    left_placeholder = col1.empty()
    right_placeholder = col2.empty()

    # Main loop to update heatmaps
    while True:
        # Create heatmap for left foot if the serial connection is available
        if ser_left and pressure_matrix_left is not None:
            fig_left = px.imshow(pressure_matrix_left, color_continuous_scale='hot',
                                 labels={'color': 'Pressure'},
                                 title="Left Foot Pressure",
                                 zmin=1, zmax=100,  # Set min and max pressure for color scale
                                 aspect='equal')  # Maintain aspect ratio
            # Set interpolation option
            fig_left.update_traces(zsmooth='best')
            fig_left.update_xaxes(title='Sensor Columns')
            fig_left.update_yaxes(title='Sensor Rows')
            left_placeholder.plotly_chart(fig_left, use_container_width=True)
        else:
            left_placeholder.empty()  # Clear the left foot heatmap if no data
        
        # Create heatmap for right foot if the serial connection is available
        if ser_right and pressure_matrix_right is not None:
            fig_right = px.imshow(pressure_matrix_right, color_continuous_scale='hot',
                                  labels={'color': 'Pressure'},
                                  title="Right Foot Pressure",
                                  zmin=1, zmax=100,  # Set min and max pressure for color scale
                                  aspect='equal')  # Maintain aspect ratio
            # Set interpolation option
            fig_right.update_traces(zsmooth='best')
            fig_right.update_xaxes(title='Sensor Columns')
            fig_right.update_yaxes(title='Sensor Rows')
            right_placeholder.plotly_chart(fig_right, use_container_width=True)
        else:
            right_placeholder.empty()  # Clear the right foot heatmap if no data

        time.sleep(0.01)  # Sleep for a short time to control the update rate

import scipy.io
import numpy as np
import pandas as pd
import os

# --- Configuration ---
file_path = 'data_cn_project_iii_a25.mat'
# Using a slightly different name to avoid confusion with previous versions
spike_matrix_csv_file = 'all_spike_times_matrix_comma_sep.csv'
stimulus_csv_file = 'stimulus.csv'
# --- ---

try:
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)
    print(f"Successfully loaded '{file_path}'")
    print("-" * 30)

    # --- Process All_Spike_Times ---
    if 'All_Spike_Times' in mat_data:
        print("Processing All_Spike_Times into comma-separated matrix format...")
        all_spikes_data = mat_data['All_Spike_Times']
        num_neurons, num_repetitions = all_spikes_data.shape

        spike_vector_list = []
        # Create the long-format list first
        for i in range(num_neurons):
            for j in range(num_repetitions):
                spike_times = all_spikes_data[i, j].flatten() # Ensure it's 1D

                # --- *** KEY CHANGE HERE *** ---
                # Convert the list/array of times into a single COMMA-separated string
                spike_times_str = ','.join(map(str, spike_times))
                # --- *** ---

                spike_vector_list.append({
                    'neuron': i + 1,
                    'repetition': j + 1,
                    'spike_times_vector_s': spike_times_str # Store the comma-separated string
                })

        # Create the long-format DataFrame
        spikes_long_df = pd.DataFrame(spike_vector_list)

        # --- Pivot the DataFrame (same as before) ---
        spikes_matrix_df = spikes_long_df.pivot(index='neuron',
                                                columns='repetition',
                                                values='spike_times_vector_s')
        spikes_matrix_df.columns = [f"{col}" for col in spikes_matrix_df.columns]

        # Save the pivoted DataFrame to CSV
        spikes_matrix_df.to_csv(spike_matrix_csv_file)
        print(f"Saved spike time matrix to '{spike_matrix_csv_file}'")
        print("Rows are neurons, columns are repetitions.")
        print("Each cell contains a COMMA-separated string of spike times.")

    else:
        print("Variable 'All_Spike_Times' not found in the .mat file.")

    print("-" * 30)

    # --- Process Stimulus (remains the same) ---
    if 'Stimulus' in mat_data:
        if not os.path.exists(stimulus_csv_file):
             print("Processing Stimulus...")
             stimulus_data = mat_data['Stimulus'].flatten()
             stimulus_df = pd.DataFrame({'stimulus_db': stimulus_data})
             stimulus_df.to_csv(stimulus_csv_file, index=False)
             print(f"Saved stimulus data to '{stimulus_csv_file}'")
        else:
             print(f"'{stimulus_csv_file}' already exists. Skipping stimulus processing.")
    else:
        print("Variable 'Stimulus' not found in the .mat file.")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
from dataclasses import dataclass
import pandas as pd
from typing import Dict, List
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import interp1d
import json
import argparse
import re

from scipy.optimize import fsolve

def plot_and_save_coef(df, df_final,channel):

    columns = ['Gain', 'Phase', 'OffsetReal', 'OffsetImaginary']
    axes_names = ['Gain [-]', 'Phase [deg]', 'OffsetReal [Ohm]', 'OffsetImaginary [Ohm]']

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Calibration coefficients', fontsize=16)

    # Save to CSV with channel prefix
    df.to_csv(f'{channel}_corr_coef_raw.csv', index=False)
    df_final.to_csv(f'{channel}_corr_coef_smooth.csv', index=False)

    for i, col in enumerate(columns):
        line, = axes[i].plot(df['Frequency'], df[col], '.', label='Raw')
        axes[i].plot(df_final['Frequency'], df_final[col], '--', label='Final', color = line.get_color())
        axes[i].set_ylabel(axes_names[i])
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Frequency [Hz]')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(channel+'_coef.png')

def smooth_coef(df):


    # Polynomial model: 3rd order
    def poly3(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    # Store fitted data
    fitted_data = {'Frequency': df['Frequency'].values}

    # Loop through each Y column and fit
    x = df['Frequency'].values
    for col in df.columns:
        if col == 'Frequency':
            continue
        y = df[col].values
        params, _ = curve_fit(poly3, x, y)
        y_fit = poly3(x, *params)
        fitted_data[col] = y_fit

    # Create new DataFrame with fitted results
    df_fitted = pd.DataFrame(fitted_data)

    return df_fitted

def find_first_higher_frequency_index(df, freq_point):
    """
    Returns the index of the first row in df['frequency'] that is greater than freq_point.
    Returns None if no such point exists.
    """
    higher_freq = df[df['Frequency'] > freq_point]
    if higher_freq.empty:
        return None
    return higher_freq.index[-1]

def convert_EIS_sweep_2_standard(df):

    # Extract the common channel from the first column
    first_col = df.columns[0]
    match = re.search(r'_CH\d+_', first_col)
    if match:
        channel = match.group(0).strip('_')  # e.g., 'CH1'

        # Rename all columns by removing the channel
        new_columns = {
            col: col.replace(match.group(0), '_')  # keep the underscores clean
            for col in df.columns
        }
    df = df.rename(columns=new_columns)
    
    df_standard = pd.DataFrame()

    df_standard['Impedance_real']      = df.Y_y_EisZRealRaw
    df_standard['Impedance_imag']      = df.Y_y_EisZImaginaryRaw
    df_standard['Impedance_amplitude'] = df.Y_y_EisZMagnitudeRaw 
    df_standard['Impedance_phase']     = df.Y_y_EisZPhaseRaw  
    df_standard['Frequency']           = df.P_x_RippleFrequency  

    return(df_standard, channel)  

def interpolate_dataframe(df, freq_col, new_freqs, kind='linear', extrapolate=True):
    """
    Interpolates numeric columns in a dataframe based on a frequency column.
    Non-numeric columns are filled with ''.

    Parameters:
        df (pd.DataFrame): Input dataframe with a frequency column.
        freq_col (str): Name of the frequency column in df.
        new_freqs (array-like): New frequency values to interpolate to.
        kind (str): Type of interpolation (e.g. 'linear', 'cubic').
        extrapolate (bool): Whether to allow extrapolation beyond the original freq range.

    Returns:
        pd.DataFrame: New dataframe with interpolated values at new_freqs.
    """
    new_data = {freq_col: new_freqs}
    x = df[freq_col].values

    for col in df.columns:
        if col == freq_col:
            continue
        try:
            y = df[col].astype(float).values
            if extrapolate:
                f = interp1d(x, y, kind=kind, fill_value="extrapolate")
            else:
                f = interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)
            new_data[col] = f(new_freqs)
        except (ValueError, TypeError):
            # Non-numeric column: fill with empty string
            new_data[col] = [''] * len(new_freqs)

    return pd.DataFrame(new_data)

def deduce_calibration_params(ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im):
    """
    Deduce calibration parameters from two reference measurements.
    
    Each reference should be a dictionary with the following keys:
        'R_ref': known (true) resistance of the reference standard (ohms)
        'X_ref': known (true) reactance of the reference standard (ohms)
        'R_meas': measured resistance (ohms)
        'X_meas': measured reactance (ohms)
    
    The measurement model is:
    Z_meas = G * [Z_ref + Z_fixture] * exp(j * phase_error),
    where:
    Z_ref = R_ref + j*X_ref   (known for each standard)
    Z_fixture = R_fixture + j*X_fixture   (unknown, same for all)
    
    Unknown calibration parameters:
        G           : gain error (scalar)
        phase_error : phase error (radians)
        R_fixture   : fixture resistance (ohms)
        X_fixture   : fixture reactance (ohms)
    
    Returns:
        G, phase_error, R_fixture, X_fixture
    """
    # Unpack reference 1 values
    R1 = ref_1_re
    X1 = ref_1_im
    R1_meas = ch_1_re
    X1_meas = ch_1_im
    
    # Unpack reference 2 values
    R2 = ref_2_re
    X2 = ref_2_im
    R2_meas = ch_2_re
    X2_meas = ch_2_im
    
    # Define the system of equations.
    # p = [G, theta, R_fixture, X_fixture]

    def equations(p):
        G, theta, R_fix, X_fix = p
        # For reference 1:
        eq1 = G * ((R1 + R_fix)*np.cos(theta) - (X1 + X_fix)*np.sin(theta)) - R1_meas
        eq2 = G * ((R1 + R_fix)*np.sin(theta) + (X1 + X_fix)*np.cos(theta)) - X1_meas
        # For reference 2:
        eq3 = G * ((R2 + R_fix)*np.cos(theta) - (X2 + X_fix)*np.sin(theta)) - R2_meas
        eq4 = G * ((R2 + R_fix)*np.sin(theta) + (X2 + X_fix)*np.cos(theta)) - X2_meas
        return [eq1, eq2, eq3, eq4]
    
    # Initial guess: assume no gain error, no phase error, and zero fixture impedance.
    p0 = [1.0, 0.0, 0.0, 0.0]
    
    solution, infodict, ier, mesg = fsolve(equations, p0, xtol=1e-12, full_output=True)
    if ier != 1:
        raise RuntimeError("Calibration parameters did not converge: " + mesg)
    
    G, theta, R_fix, X_fix = solution

    return G, theta*180/np.pi, R_fix, X_fix

def deduce_calibration_params_no_fixture(ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im):
    """
    Deduce calibration parameters from two reference measurements,
    assuming that fixture impedance is zero (i.e. R_fixture = 0 and X_fixture = 0).
    
    Each reference is a dictionary with:
        'R_ref' : known (true) resistance of the reference standard (ohms)
        'X_ref' : known (true) reactance of the reference standard (ohms)
        'R_meas': measured resistance (ohms)
        'X_meas': measured reactance (ohms)
    
    The measurement model is:
        Z_meas = G * Z_ref * exp(j * theta),
    where Z_ref = R_ref + j*X_ref.
    
    Unknown calibration parameters:
        G       : gain error (scalar)
        theta   : phase error (radians)
    
    The following equations are used for each reference:
        For reference 1:
        (1) G*(R1*cos(theta) - X1*sin(theta)) = R1_meas
        (2) G*(R1*sin(theta) + X1*cos(theta)) = X1_meas
        For reference 2:
        (3) G*(R2*cos(theta) - X2*sin(theta)) = R2_meas
        (4) G*(R2*sin(theta) + X2*cos(theta)) = X2_meas
    
    These 4 equations form an overdetermined system for the 2 unknowns,
    which is solved in a least-squares sense.
    
    Returns:
        G, theta
    """
    def residuals(p, ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im):
        G, theta = p
        
        # Unpack reference 1:
        # Unpack reference 1 values
        R1 = ref_1_re
        X1 = ref_1_im
        R1_meas = ch_1_re
        X1_meas = ch_1_im
        
        # Unpack reference 2 values
        R2 = ref_2_re
        X2 = ref_2_im
        R2_meas = ch_2_re
        X2_meas = ch_2_im
        
        # Equations for reference 1:
        eq1 = G * (R1 * np.cos(theta) - X1 * np.sin(theta)) - R1_meas
        eq2 = G * (R1 * np.sin(theta) + X1 * np.cos(theta)) - X1_meas
        
        # Equations for reference 2:
        eq3 = G * (R2 * np.cos(theta) - X2 * np.sin(theta)) - R2_meas
        eq4 = G * (R2 * np.sin(theta) + X2 * np.cos(theta)) - X2_meas
        
        return [eq1, eq2, eq3, eq4]
    
    # Initial guess: G=1, theta=0
    p0 = [1.0, 0.0]
    
    result = least_squares(residuals, p0, args=(ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im))
    G, theta = result.x
    return G, theta

def deduce_calibration_params_overdefined(ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im):
    """
    Deduce calibration parameters from two reference measurements, 
    assuming R_fixture = 0, by solving an overdetermined system (4 equations, 3 unknowns)
    in a least-squares sense.

    Each reference is a dictionary with the following keys:
        'R_ref' : known (true) resistance of the reference standard (ohms)
        'X_ref' : known (true) reactance of the reference standard (ohms)
        'R_meas': measured resistance (ohms)
        'X_meas': measured reactance (ohms)
    
    The measurement model is:
        Z_meas = G * [Z_ref + Z_fixture] * exp(j * theta),
    with:
        Z_ref    = R_ref + j*X_ref
        Z_fixture = 0 + j*X_fixture   (since R_fixture = 0)

    Unknown calibration parameters:
        G           : gain error (scalar)
        theta       : phase error (radians)
        X_fixture   : fixture reactance (ohms)

    Returns:
        G, theta, X_fixture
    """
    def residuals(p, ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im):
        G, theta, X_fix = p
        # Unpack reference 1:
        # Unpack reference 1 values
        R1 = ref_1_re
        X1 = ref_1_im
        R1_meas = ch_1_re
        X1_meas = ch_1_im
        
        # Unpack reference 2 values
        R2 = ref_2_re
        X2 = ref_2_im
        R2_meas = ch_2_re
        X2_meas = ch_2_im
        
        # Equation (1) for ref1 (real part)
        eq1 = G * (R1 * np.cos(theta) - (X1 + X_fix) * np.sin(theta)) - R1_meas
        # Equation (2) for ref1 (imaginary part)
        eq2 = G * (R1 * np.sin(theta) + (X1 + X_fix) * np.cos(theta)) - X1_meas
        # Equation (3) for ref2 (real part)
        eq3 = G * (R2 * np.cos(theta) - (X2 + X_fix) * np.sin(theta)) - R2_meas
        # Equation (4) for ref2 (imaginary part)
        eq4 = G * (R2 * np.sin(theta) + (X2 + X_fix) * np.cos(theta)) - X2_meas
        return [eq1, eq2, eq3, eq4]
    
    # Initial guess: assume ideal gain, zero phase error, and zero fixture reactance.
    p0 = [1.0, 0.0, 0.0]
    
    # Solve the overdetermined system in a least-squares sense.
    result = least_squares(residuals, p0, args=(ref_1_re, ref_1_im, ch_1_re, ch_1_im, ref_2_re, ref_2_im, ch_2_re, ch_2_im))
    
    G, theta, X_fix = result.x
    return G, theta*180/np.pi, X_fix

def deduce_calibration_params_single_ref(ref_1_re, ref_1_im, ch_1_re, ch_1_im):
    """
    Deduce calibration parameters from a single reference measurement,
    assuming the measurement model:
    
        Z_meas = G * Z_ref * exp(j * theta)
    
    where:
        Z_ref  = R_ref + j*X_ref   (known true impedance)
        Z_meas = R_meas + j*X_meas  (measured impedance)
    
    The unknown calibration parameters are:
        G      : gain error (scalar)
        theta  : phase error (radians)
    
    They can be computed as:
        ratio = Z_meas / Z_ref,
        G = |ratio|, and
        theta = angle(ratio).
    
    Parameters:
        ref : dict with keys:
            'R_ref'  - true resistance of the reference (ohms)
            'X_ref'  - true reactance of the reference (ohms)
            'R_meas' - measured resistance (ohms)
            'X_meas' - measured reactance (ohms)
    
    Returns:
        G, theta
    """
    R1 = ref_1_re
    X1 = ref_1_im
    R1_meas = ch_1_re
    X1_meas = ch_1_im
    
    Z_ref = R1 + 1j * X1
    Z_meas = R1_meas + 1j * X1_meas
    
    if Z_ref == 0:
        raise ValueError("Reference impedance is zero; cannot compute calibration parameters.")
    
    ratio = Z_meas / Z_ref
    G = np.abs(ratio)
    theta = np.angle(ratio)
    return G, theta

def update_json_with_df(df, json_file):
    try:
        # Read the existing JSON data
        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Make sure the structure is correct
        if "EIS_Calibration" not in json_data or "Frequencies" not in json_data["EIS_Calibration"]:
            print("❌ Invalid JSON structure!")
            return

        # Get the existing frequencies list from the JSON
        existing_frequencies = json_data["EIS_Calibration"]["Frequencies"]

        # Convert existing frequencies to a dictionary for quick lookup by Frequency_Point
        freq_dict = {freq["Frequency_Point"]: freq for freq in existing_frequencies}

        # Update the frequencies list in the JSON with data from the dataframe (matching by Frequency_Point)
        for i, row in df.iterrows():
            frequency_point = row["Frequency_Point"]

            # Update frequency data only if Frequency_Point exists in the JSON
            if frequency_point in freq_dict:
                #freq_dict[frequency_point]["Frequency"] = row["Frequency"]
                freq_dict[frequency_point]["Gain"] = row["Gain"]
                freq_dict[frequency_point]["Phase"] = row["Phase"]
                freq_dict[frequency_point]["OffsetReal"] = row["OffsetReal"]
                freq_dict[frequency_point]["OffsetImaginary"] = row["OffsetImaginary"]

        # Convert the updated dictionary back to a list
        updated_frequencies = list(freq_dict.values())

        # Update the frequencies in the JSON data
        json_data["EIS_Calibration"]["Frequencies"] = updated_frequencies

        # Write updated JSON data to a new file
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=3)
        print(f"✅ JSON updated and saved to: {json_file}")
    
    except Exception as e:
        print(f"❌ Error updating JSON: {e}")

def update_json_with_df_new(df, default_file, gnerated_file):

    # Load the JSON file
    with open(default_file, "r") as f:
        json_data = json.load(f)

    # Extract Frequency_Point values from JSON
    frequency_list = [
        item["value"]
        for item in json_data["EIS_default_calibration"]
        if item["name"].startswith("Frequency") and not item["name"].startswith("Frequency_Point")
    ]

    # Start with initial fixed entries
    json_list = [
        {"name": "VersionInfo", "value": 1.0, "type": "d"},
        {"name": "Debug", "value": 0.0, "type": "d"}
    ]

    # Add rows from DataFrame
    for i, row in df.iterrows():
        idx = i + 1
        json_list.extend([
            {"name": f"Frequency_Point{idx}", "value": float(idx), "type": "d"},
            {"name": f"Frequency{idx}", "value": row['Frequency'], "type": "d"},
            {"name": f"Gain{idx}", "value": row['Gain'], "type": "d"},
            {"name": f"Phase{idx}", "value": row['Phase'], "type": "d"},
            {"name": f"OffsetReal{idx}", "value": row['OffsetReal'], "type": "d"},
            {"name": f"OffsetImaginary{idx}", "value": row['OffsetImaginary'], "type": "d"}
        ])

    # Add the remaining entries using frequency points beyond the DataFrame
    df_len = len(df)
    last_row = df.iloc[-1]

    for i, freq_index in enumerate(frequency_list[df_len:]):
        idx = df_len + i + 1
        json_list.extend([
            {"name": f"Frequency_Point{idx}", "value": float(idx), "type": "d"},
            {"name": f"Frequency{idx}", "value": freq_index, "type": "d"},  # use the stored frequency
            {"name": f"Gain{idx}", "value": last_row['Gain'], "type": "d"},
            {"name": f"Phase{idx}", "value": 0, "type": "d"},
            {"name": f"OffsetReal{idx}", "value": last_row['OffsetReal'], "type": "d"},
            {"name": f"OffsetImaginary{idx}", "value": 0, "type": "d"}
        ])

    # Final output
    final_output = {gnerated_file: json_list}

    # Optionally write to file
    with open(gnerated_file+'.json', "w") as f:
        json.dump(final_output, f, indent=4)

    print("JSON written successfully.")


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Calculate EIS calibration JSON.")
    parser.add_argument("ref1_filename",  help="Path to the input CSV file")
    parser.add_argument("ref2_filename",  help="Path to the input CSV file")
    parser.add_argument("eis1_filename",  help="Path to the input CSV file")
    parser.add_argument("eis2_filename",  help="Path to the input CSV file")
    parser.add_argument("default_file",   help="Input JSON file name")
    parser.add_argument("smooth",         help="Is the soothing applayed")
    parser.add_argument("isReport",       help="Should the report be printed")
    
    try:
        args, _ = parser.parse_known_args()
    except:
        print("No command-line arguments detected. Using default file for IDE/debugging mode.")
        args = argparse.Namespace()
        args.ref1_filename = "Ref_1mOhm.csv"  # ← Change this to your test file
        args.ref2_filename = "Ref_500uOhm.csv" 
        args.eis1_filename = "CH1_1mOhm_1.csv" 
        args.eis2_filename = "CH1_500uOhm_1.csv" 
        args.default_file  = "EIS_default_calibration.json"
        args.smooth        = 1
        args.isReport      = 1

    # Load the CSVs
    try:
        # Import ref impedance Ref files
        ref1_df = pd.read_csv(args.ref1_filename)
        ref2_df = pd.read_csv(args.ref2_filename)

        # Import ref impedance EIS sweep files
        #eis1_df = pd.read_csv(args.eis1_filename, skiprows=0, delimiter=',')
        #eis2_df = pd.read_csv(args.eis2_filename, skiprows=0, delimiter=',')

        eis1_df = pd.read_csv(args.eis1_filename, skiprows=11, delimiter='\t')
        eis2_df = pd.read_csv(args.eis2_filename, skiprows=11, delimiter='\t')

        eis1_df, channel = convert_EIS_sweep_2_standard(eis1_df)
        eis2_df, channel = convert_EIS_sweep_2_standard(eis2_df)

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    last_ref_freq = ref1_df['Frequency'][0]
    freq_index = find_first_higher_frequency_index(eis1_df, last_ref_freq)

    eis1_df = eis1_df.iloc[:freq_index+1]
    eis2_df = eis2_df.iloc[:freq_index+1]

    eis1_df.rename(columns={'Point number': 'Frequency_Point'}, inplace=True)
    eis2_df.rename(columns={'Point number': 'Frequency_Point'}, inplace=True)

    # For each frequencey point deduce calibration parametres and write them in the calibration file
    ref1_interp = interpolate_dataframe(ref1_df,  'Frequency',  eis1_df.Frequency, kind='linear', extrapolate=True)
    ref2_interp = interpolate_dataframe(ref2_df,  'Frequency',  eis2_df.Frequency, kind='linear', extrapolate=True)

    df = pd.DataFrame(columns=['Frequency', 'Gain', 'Phase', 'OffsetReal', 'OffsetImaginary'])

    for ref_1, ref_2, meas_1, meas_2 in zip(ref1_interp.itertuples(index=False),ref2_interp.itertuples(index=False),
                                                eis1_df.itertuples(index=False),   eis2_df.itertuples(index=False)):

        G, theta, R_fix, X_fix = deduce_calibration_params(ref_1.Impedance_real,     ref_1.Impedance_imag, 
                                                           meas_1.Impedance_real,    meas_1.Impedance_imag,  
                                                           ref_2.Impedance_real,     ref_2.Impedance_imag, 
                                                           meas_2.Impedance_real,    meas_2.Impedance_imag)

        df.loc[len(df)] =  [meas_1.Frequency, G, theta, R_fix, X_fix]

    if args.smooth:
        df_final= smooth_coef(df)        
        update_json_with_df_new(df_final, args.default_file, "EIS_calibration_"+channel)
    else:   
        df_final = df
        update_json_with_df_new(df_final, args.default_file, "EIS_calibration_"+channel)
    
    if args.isReport:
        plot_and_save_coef(df, df_final,channel)


if __name__ == "__main__":
    main()  # Manual params here

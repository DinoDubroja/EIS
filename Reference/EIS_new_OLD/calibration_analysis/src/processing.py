from dataclasses import dataclass
import pandas as pd
from typing import Dict, List
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from copy import deepcopy
from scipy.optimize import fsolve
from enum import Enum
import re
from scipy.optimize import curve_fit

matplotlib.rcParams['lines.linewidth'] = 0.8
#from processing import Ac_correction

class Type(Enum):
    EIS_sweep = 1
    EIS_sweep_2 = 3
    Ref = 2

class Name(Enum):
    Legacy = 1
    Simple = 2

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

@dataclass
class Eis_sweep:
    file_name: str
    channel = str
    data: pd.DataFrame
    currents: List
    impedance_value: str
    goertzel_config: str
    date: str
    run_number: int
    comment: str
    corrected: str

    def __init__(self, file_path: str, type = Type.EIS_sweep, name= Name.Legacy):
          
        self.file_name = file_path.__str__()
        split_list_1 = (self.file_name).split('.')[0].split('\\')
        split_list_2 = split_list_1[-1].split('_')

        if type==Type.EIS_sweep or type==Type.EIS_sweep_2:
            
            if name == Name.Legacy:

                self.impedance_value = split_list_1[1]
                self.channel = split_list_2[2]
                self.goertzel_config = split_list_2[3]
                self.date = split_list_2[-1]

                if len(split_list_2) >= 6:
                    self.run_number = int(split_list_2[4])
                else:
                    self.run_number = 1

            elif name == Name.Simple:

                self.impedance_value = split_list_1[1]
                self.channel = split_list_2[2]
                self.run_number = int(split_list_2[3])
            
            if type==Type.EIS_sweep:
                self.data = pd.read_csv(file_path, skiprows=0, delimiter=',')
            elif type==Type.EIS_sweep_2:
                self.data = pd.read_csv(file_path, skiprows=11, delimiter='\t')

        elif type==Type.Ref:

            self.impedance_value = split_list_1[1]
            self.channel = split_list_2[3]

            if len(split_list_2) == 5:
                self.run_number = int(split_list_2[4])
            else:
                self.run_number = 1

            self.data = pd.read_csv(file_path)

        self.correcion = 'None'


        if type==Type.EIS_sweep:
            self.data = convert_EIS_sweep_2_standard(self.data)
        elif type==Type.EIS_sweep_2:
            self.data = convert_EIS_2_sweep_2_standard(self.data)
        elif type==Type.Ref:
            self.data = convert_ref_2_standard(self.data)

        self.data = unwrap_phase(self.data)

    def apply_dc_correction(self, gain):

        self.data.Impedance_real      = self.data.Impedance_real*gain
        self.data.Impedance_imag      = self.data.Impedance_imag*gain
        self.data.Impedance_amplitude = self.data.Impedance_amplitude*gain
        self.data.Impedance_phase     = self.data.Impedance_phase

    def apply_ac_correction(self, ac_corr):

        df_meas = pd.DataFrame()

        df_meas['Frequency'] = self.data.Frequency
        df_meas['R_meas']    = self.data.Impedance_real
        df_meas['X_meas']    = self.data.Impedance_imag
        
        # Merge the two DataFrames on the 'Frequency' column.
        ac_corr_interp = interpolate_dataframe(ac_corr.corr_coef_df,  'Frequency', df_meas.Frequency, kind='linear', extrapolate=True)
        df_merged = df_meas.join(ac_corr_interp, rsuffix='_comp')

        # Combine the measured real and imaginary parts into a complex number.
        df_merged['Z_meas'] = df_merged['R_meas'] + 1j * df_merged['X_meas']

        # Apply the correction function row-by-row.
        df_merged['Z_ref'] = df_merged.apply(
            lambda row: Eis_sweep.correct_impedance(
                row['Z_meas'],
                row['G'],
                row['theta'],
                row['R_fix'],
                row['X_fix']
            ),
            axis=1
        )

        self.data.Impedance_real      = df_merged['Z_ref'].apply(np.real)
        self.data.Impedance_imag      = df_merged['Z_ref'].apply(np.imag)
        self.data.Frequency           = df_merged['Frequency']
        self.data.Impedance_amplitude = df_merged['Z_ref'].apply(np.abs)
        self.data.Impedance_phase     = df_merged['Z_ref'].apply(np.angle)*180/np.pi


    def correct_impedance(Z_meas, G, phase_error, R_fixture, X_fixture):
        """
        Correct the measured impedance Z_meas using the calibration parameters.
        
        The measurement model is:
        Z_meas = G * [Z_ref + (R_fixture + j*X_fixture)] * exp(j*phase_error),
        where Z_ref is the true impedance of the device under test.
        
        Solve for Z_ref:
        Z_ref = Z_meas/(G*exp(j*phase_error)) - (R_fixture + j*X_fixture)
        
        Parameters:
            Z_meas     : Measured impedance (complex number)
            G          : Gain error (scalar)
            phase_error: Phase error (deg)
            R_fixture  : Fixture resistance (ohms)
            X_fixture  : Fixture reactance (ohms)
        
        Returns:
            Z_ref      : Corrected (true) impedance (complex number)
        """

        a = (G * np.exp(1j * phase_error*np.pi/180))
        b = (R_fixture + 1j * X_fixture)
        Z_ref = Z_meas / (G * np.exp(1j * phase_error*np.pi/180)) - (R_fixture + 1j * X_fixture)

        return Z_ref

class Ref_impedance:
    file_name: str
    data: pd.DataFrame
    impedance_value: int
    comment: str

    def __init__(self, file_path: str):
        
        self.file_name = os.path.basename(file_path)
        split_list = (self.file_name).split('.')[0].split('_')

        self.impedance_value = split_list[0]+'_'+split_list[1]
        self.number = split_list[2]

        self.data = pd.read_csv(file_path, sep=',')
        self.data = convert_ref_2_standard(self.data)
        self.data = unwrap_phase(self.data)

        self.channel = 'Ref'

def unwrap_phase(df):
    df.Impedance_phase = 180/np.pi*np.unwrap(df.Impedance_phase*np.pi/180, period=np.pi*2, discont=np.pi/2)
    df.Impedance_phase = (df.Impedance_phase + 180) % 360 - 180
    df['Impedance_real'] = df.Impedance_amplitude * np.cos(df.Impedance_phase*np.pi/180)
    df['Impedance_imag'] = df.Impedance_amplitude * np.sin(df.Impedance_phase*np.pi/180)

    return df

def convert_ref_2_standard(df):
    
    df_standard = pd.DataFrame()

    df_standard['Impedance_real']      = df.Impedance_real
    df_standard['Impedance_imag']      = df.Impedance_imag
    df_standard['Impedance_amplitude'] = df.Impedance_amplitude 
    df_standard['Impedance_phase']     = df.Impedance_phase  
    df_standard['Frequency']           = df.Frequency  

    return(df_standard)  

def convert_EIS_sweep_2_standard(df):
    
    df_standard = pd.DataFrame()

    df_standard['Impedance_real']      = df.Y_y_EIS_ZReal_raw
    df_standard['Impedance_imag']      = df.Y_y_EIS_ZImaginary_raw
    df_standard['Impedance_amplitude'] = df.Y_y_EIS_ZMagnitude_raw 
    df_standard['Impedance_phase']     = df.Y_y_EIS_ZPhase_raw_degree  
    df_standard['Frequency']           = df.P_x_Goertzel_Frequency  

    return(df_standard)  

def convert_EIS_2_sweep_2_standard(df):

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

    return(df_standard)  

        

class Ac_correction:

    corr_coef_df: pd.DataFrame

    def __init__(self, ref_impedance_1: Ref_impedance, ref_impedance_2: Ref_impedance, eis_sweep_1: Eis_sweep, eis_sweep_2: Eis_sweep, smooth=False):

        eis1_df = deepcopy(eis_sweep_1.data) # eis_sweep_1.data.iloc[:10]
        eis2_df = deepcopy(eis_sweep_2.data) # eis_sweep_2.data.iloc[:10]

        eis1_df.rename(columns={'Point number': 'Frequency_Point'}, inplace=True)
        eis2_df.rename(columns={'Point number': 'Frequency_Point'}, inplace=True)

        ref1_interp = interpolate_dataframe(ref_impedance_1.data,  'Frequency',  eis1_df.Frequency, kind='linear', extrapolate=True)
        ref2_interp = interpolate_dataframe(ref_impedance_2.data,  'Frequency',  eis2_df.Frequency, kind='linear', extrapolate=True)

        df = pd.DataFrame(columns=['Frequency', 'G', 'theta', 'R_fix', 'X_fix'])
        
        for ref_1, ref_2, meas_1, meas_2 in zip(ref1_interp.itertuples(index=False), ref2_interp.itertuples(index=False),
                                                    eis1_df.itertuples(index=False),      eis2_df.itertuples(index=False)):

            G, theta, R_fix, X_fix = Ac_correction.deduce_calibration_params(ref_1.Impedance_real, ref_1.Impedance_imag, 
                                                                             meas_1.Impedance_real, meas_1.Impedance_imag,  
                                                                             ref_2.Impedance_real, ref_2.Impedance_imag, 
                                                                             meas_2.Impedance_real, meas_2.Impedance_imag)
            
           
            #G, theta, X_fix = Ac_correction.deduce_calibration_params_overdefined(ref_1.Impedance_real, ref_1.Impedance_imag, 
            #                                                                 meas_1.Y_y_EIS_ZReal_raw, meas_1.Y_y_EIS_ZImaginary_raw,  
            #                                                                 ref_2.Impedance_real, ref_2.Impedance_imag, 
            #                                                                 meas_2.Y_y_EIS_ZReal_raw, meas_2.Y_y_EIS_ZImaginary_raw)
            
            #G, theta = Ac_correction.deduce_calibration_params_single_ref(ref_1.Interpolated_R_ref, ref_1.Interpolated_X_ref, 
            #                                                              ref_1.Y_y_EIS_ZReal_raw, ref_1.Y_y_EIS_ZImaginary_raw)
            
            df.loc[len(df)] = [ref_1.Frequency, G, theta, R_fix, X_fix]

        if smooth==True:
            df_fitted = Ac_correction.smooth_coef(df)        
            self.corr_coef_raw_df = df.sort_values('Frequency', ascending=0)
            self.corr_coef_df = df_fitted.sort_values('Frequency', ascending=0)
        else:
            self.corr_coef_raw_df = df.sort_values('Frequency', ascending=0)
            self.corr_coef_df =     df.sort_values('Frequency', ascending=0)

        self.corr_coef_raw_df.reset_index(drop=True, inplace=True)
        self.corr_coef_df.reset_index(drop=True, inplace=True)


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

# Plotting

def plot_preferences(axs, xlabel = '', ylabel = '', title = '', legend = False):
        
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        if legend: axs.legend()
        axs.grid()

        axs.set_title(title, fontweight="bold")

        axs.minorticks_on()
        axs.grid(which='minor', linewidth = 0.5, linestyle='dotted')
        axs.grid(which='major', linewidth = 0.5, linestyle='-')

def plot_bode(ax, data, label, title, only_HF=False):

    if only_HF: data = data[0:11]

    ax[0].plot(data.Frequency, data.Impedance_amplitude*1000, label=label)
    ax[1].plot(data.Frequency, data.Impedance_phase, label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Impedance [mOhm]', title = title, legend=True)
    plot_preferences(ax[1], xlabel = 'Frequency [Hz]', ylabel = 'Phase [deg]',     title = '', legend=True)

def plot_nyquist(ax, data, label, title, only_HF=False):

    if only_HF: data = data[0:11]

    ax.plot(data.Impedance_real*1000, data.Impedance_imag*1000, '.-', label=label)
    plot_preferences(ax, xlabel = 'Re [mOhm]', ylabel = 'Im [mOhm]', title = title, legend=True)

def plot_nyquist_ref(ax, data, label, title):

    ax.plot(data['Impedance_real']*1000, data['Impedance_imag']*1000, '.-', color='black', linewidth=1.5, label=label)
    plot_preferences(ax, xlabel = 'Re [mOhm]', ylabel = 'Im [mOhm]', title = title, legend=True)

def plot_bode_ref(ax, data, label, title):

    ax[0].plot(data.Frequency, data.Impedance_amplitude*1000, color='black', linewidth=1.5, label=label)
    ax[1].plot(data.Frequency, data.Impedance_phase, color='black', linewidth=1.5, label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Impedance [mOhm]', title = title, legend=True)
    plot_preferences(ax[1], xlabel = 'Frequency [Hz]', ylabel = 'Phase [deg]',     title = '', legend=True)

def plot_bode_comparion(ax, data_ch, data_ref, label, title):

    data = pd.DataFrame()

    data_ch = data_ch.loc[data_ref.index]

    data_ref_interp = interpolate_dataframe(data_ref,  'Frequency', data_ch.Frequency, kind='linear', extrapolate=True)

    data['Frequency'] = data_ref_interp.Frequency
    data['Amplitude_diff_abs'] = data_ch.Impedance_amplitude    - data_ref_interp.Impedance_amplitude
    data['Phase_diff']         = data_ch.Impedance_phase - data_ref_interp.Impedance_phase    
    data['Amplitude_diff_rel'] = data.Amplitude_diff_abs/data_ref_interp.Impedance_amplitude*100

    ax[0].plot(data.Frequency, data.Amplitude_diff_abs*1000, label=label)
    ax[1].plot(data.Frequency, data.Amplitude_diff_rel, label=label) 
    ax[2].plot(data.Frequency, data.Phase_diff, label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Impedance error [mOhm]', title = title, legend=True)
    plot_preferences(ax[1], xlabel = '', ylabel = 'Impedance error [%]',     title = '', legend=True)
    plot_preferences(ax[2], xlabel = 'Frequency [Hz]', ylabel = 'Phase error [deg]', title = '', legend=True)

def plot_bode_re_imag(ax,data, label, title, only_HF=False):

    if only_HF: data = data[0:11]

    ax[0].plot(data.Frequency, data.Impedance_real*1000, '.-', label=label)
    ax[1].plot(data.Frequency, data.Impedance_imag*1000,'.-', label=label) 
    ax[2].plot(data.Frequency, data.Impedance_imag/(2*math.pi*data.P_x_Goertzel_Frequency)*1000000000, '.-', label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Real [mOhm]', title=title, legend=True)
    plot_preferences(ax[1], xlabel = '', ylabel = 'Imag [mOhm]', legend=True)
    plot_preferences(ax[2], xlabel = 'Frequency [Hz]', ylabel = 'Inductance [nH]', legend=True)

def plot_muliple_bode(measurements, impedance_value, title, references={}, file_name=None, only_HF = False):

    fig, ax = plt.subplots(2, figsize=(6, 9))

    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue
        
        label =  str(measurement.channel) + '_' +str(measurement.run_number)
        plot_bode(ax, measurement.data, label=label, title=title, only_HF=only_HF)

    if impedance_value in references:
        plot_bode_ref(ax, references[impedance_value].data, label=references[impedance_value].channel, title=title)

    if file_name: fig.savefig(file_name, bbox_inches='tight')

def plot_muliple_nyquist(measurements, impedance_value, title, references={}, file_name=None, only_HF = False):

    fig, ax = plt.subplots(figsize=(6, 6))

    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue
        
        label =  str(measurement.channel) +'_' +str(measurement.run_number)
        plot_nyquist(ax, measurement.data, label=label, title=title, only_HF=only_HF)

    if impedance_value in references:
        plot_nyquist_ref(ax, references[impedance_value].data, label=references[impedance_value].channel, title=title)

    if file_name: fig.savefig(file_name, bbox_inches='tight')

def plot_muliple_bode_re_imag(measurements, impedance_value, title, file_name=None, only_HF = False):

    fig, ax = plt.subplots(3,figsize=(6, 9))

    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue

        label =  str(measurement.channel) + '_' + str(measurement.goertzel_config) + '_' +str(measurement.run_number)
        plot_bode_re_imag(ax, measurement.data, label=label, title=title, only_HF=only_HF)

    if file_name: fig.savefig(file_name, bbox_inches='tight')

def plot_mulitple_bode_comparion(measurements, references, impedance_value, title, file_name=None):

    fig, ax = plt.subplots(3, figsize=(6, 12))

    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue

        label =  str(measurement.channel) + '_' + str(measurement.run_number)
        plot_bode_comparion(ax, measurement.data, references[impedance_value].data, label, title)

    if file_name: fig.savefig(file_name, bbox_inches='tight')

def get_std_mean_values(measurements, impedance_value, impedance_type):

    combined = pd.DataFrame()

    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue

        if 'Frequency' in measurement.data.columns:
            df = measurement.data.set_index('Frequency')
        combined[measurement.channel+str(measurement.run_number)] = df[impedance_type]

        # Step 3: Compute standard deviation across measurements at each frequency
        std_series = combined.std(axis=1)
        mean_series = combined.mean(axis=1)

    return(std_series, mean_series)


def plot_multiple_std_dev(measurements, impedance_value, title, file_name=None):

    fig, ax = plt.subplots(2, figsize=(6, 9))

    std_series, mean_series = get_std_mean_values(measurements, impedance_value, "Impedance_amplitude")
    std_series_deg, mean_series_deg = get_std_mean_values(measurements, impedance_value, "Impedance_phase")

    ax[0].plot(std_series.index, std_series.values*2/mean_series.values*100, '.-', label=f'Std (2 sigma) of Impedance_amplitude')
    ax[1].plot(std_series_deg.index, std_series_deg.values, '.-', label=f'Std (2 sigma) of Impedance_phase') 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Impedance [%]', title = title, legend=True)
    plot_preferences(ax[1], xlabel = 'Frequency [Hz]', ylabel = 'Phase [deg]',     title = '', legend=True)

    if file_name: fig.savefig(file_name, bbox_inches='tight')

def save_data(measurements, impedance_value, goertzel_config, references={}, file_name=None):

    df_full = pd.DataFrame()
    df_list = []
    for measurement in measurements:

        if measurement.impedance_value != impedance_value:
            continue

        if measurement.goertzel_config != goertzel_config:
            continue

        if measurement.run_number != 2:
            continue
        
        label =  str(measurement.channel) + '_' +str(measurement.run_number)

        df = measurement.data[['Frequency','Impedance_amplitude','Impedance_phase','Impedance_real', 'Impedance_imag']]
        df_copy = df.copy()
        df_copy.columns = [f'{label}_{col}' for col in df_copy.columns]
        df_list.append(df_copy)

    df_full = pd.concat(df_list, axis=1)
    if file_name: df_full.to_csv(file_name +'_'+ impedance_value + '.csv', index=False)

#def create_Re_gain_list(data_ref, data_ch):
#
#    data_ref_sorted = data_ref.sort_values(by='f/Hz', ascending=False)
#    data_ref_sorted = data_ref_sorted.iloc[-5:]
#    gain_list_dict = data_ref_sorted['Re(Z)']-data_ch['Y_y_EIS_ZReal_raw']
#    gain_list_dict = gain_list_dict.fillna(0)
#    return gain_list_dict

def iterate_dict(d):
    """Recursively iterate through all values of a multi-dimensional dictionary."""
    if isinstance(d, dict):  # If it's a dictionary, go deeper
        for key, value in d.items():
            iterate_dict(value)  # Recursively process the value
    else:
        print(d)  # Print or process the value
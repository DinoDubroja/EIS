def generate_sine_wave(
    frequency: float,
    amplitude: float,
    sampling_rate: float,
    number_of_samples: int,
    phase_in: float = 0.0,
):
    """Generates a sine wave with a specified phase.

    Args:
        frequency: Specifies the frequency of the sine wave.
        amplitude: Specifies the amplitude of the sine wave.
        sampling_rate: Specifies the sampling rate of the sine wave.
        number_of_samples: Specifies the number of samples to generate.
        phase_in: Specifies the phase of the sine wave in radians.

    Returns:
        Indicates a tuple containing the generated data and the phase
        of the sine wave after generation.
    """
    duration_time = number_of_samples / sampling_rate
    duration_radians = duration_time * 2 * np.pi
    phase_out = (phase_in + duration_radians) % (2 * np.pi)
    t = np.linspace(phase_in, phase_in + duration_radians, number_of_samples, endpoint=False)

    return (amplitude * np.sin(frequency * t), phase_out)

def fft_sine(t, y):
    N = len(t)
    dt = t[1] - t[0]
    freq = fftfreq(N, dt)
    Y = fft(y.values-np.mean(y.values))
    
    # Consider only positive frequencies
    positive_idx = np.where(freq > 0)
    Y_positive = Y[positive_idx]
    freq_positive = freq[positive_idx]
    
    # Find the dominant frequency component
    idx = np.argmax(np.abs(Y_positive))
    f_est = freq_positive[idx]
    A_est = (2 * np.abs(Y_positive[idx])) / N
    phi_est = np.angle(Y_positive[idx])
    C_est = np.mean(y)
    phase_deg = (phi_est / (2 * np.pi)) * 360 % 360
    
    # Add ALIGNED flag if desired (for example, checking if phase is within a tolerance)
    # Here we use a dummy condition; modify as needed for your application.
    aligned = "YES" if abs(phase_deg - 45) < 10 else "NO"
    
    result = pd.Series({
        'Amplitude': A_est,
        'Frequency': f_est,
        'Phase': phase_deg,
        'Offset': C_est,
        'ALIGNED': aligned
    })
    
    return result

def create_df_with_prefix(series1, series2, prefix1, prefix2):
    """
    Combine two Series with prefixes into one row DataFrame
    """
    # Rename series with prefix
    series1 = series1.add_prefix(f"{prefix1}_")
    series2 = series2.add_prefix(f"{prefix2}_")

    # Merge both series into one row DataFrame
    return pd.concat([series1, series2]).to_frame().T

def import_csv_folder(folder_path):
    csv_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            name = os.path.splitext(filename)[0]
            csv_dict[name] = pd.read_csv(os.path.join(folder_path, filename))
    return csv_dict

def plot_frequency_subplots(freq_dict):
    num_freqs = len(freq_dict)
    fig, axs = plt.subplots(num_freqs, 1, figsize=(12, 4 * num_freqs), sharex=False)
    
    if num_freqs == 1:
        axs = [axs]  # Make sure axs is iterable if only one frequency
    
    for i, (freq, df) in enumerate(freq_dict.items()):
        ax1 = axs[i]
        ax2 = ax1.twinx()
        
        # Plot Voltage over Time (left axis)
        ax1.plot(df['Time'], df['Voltage'], color='b', label=f'Voltage @ {freq} Hz')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title(f'Voltage and Current over Time @ {freq} Hz')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # Plot Current over Time (right axis)
        ax2.plot(df['Time'], df['Current'], color='r', label=f'Current @ {freq} Hz')
        ax2.set_ylabel('Current (A)')
        ax2.legend(loc='upper right')
        
        ax1.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

def plot_nyquist(ax, data, label, title):

    ax.plot(data.Impedance_real*1000, data.Impedance_imag*1000, '.-', label=label)
    plot_preferences(ax, xlabel = 'Re [mOhm]', ylabel = 'Im [mOhm]', title = title, legend=True)

def plot_bode_re_imag(ax,data, label, title):

    ax[0].plot(data.Frequency, data.Impedance_real*1000, '.-', label=label)
    ax[1].plot(data.Frequency, data.Impedance_imag*1000,'.-', label=label) 
    ax[2].plot(data.Frequency, data.Impedance_imag/(2*np.pi*data.Frequency)*1000000000, '.-', label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Real [mOhm]', title=title, legend=True)
    plot_preferences(ax[1], xlabel = '', ylabel = 'Imag [mOhm]', legend=True)
    plot_preferences(ax[2], xlabel = 'Frequency [Hz]', ylabel = 'Inductance [nH]', legend=True)

def plot_bode(ax,data, label, title):

    ax[0].plot(data.Frequency, data.Impedance_real*1000, '.-', label=label)
    ax[1].plot(data.Frequency, data.Impedance_imag*1000,'.-', label=label) 

    plot_preferences(ax[0], xlabel = '', ylabel = 'Amplitude [mOhm]', title=title, legend=False)
    plot_preferences(ax[1], xlabel = '', ylabel = 'Phase [deg]', legend=False)

def plot_preferences(axs, xlabel = '', ylabel = '', title = '', legend = False):
        
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        if legend: axs.legend()
        axs.grid()

        axs.set_title(title, fontweight="bold")

        axs.minorticks_on()
        axs.grid(which='minor', linewidth = 0.5, linestyle='dotted')
        axs.grid(which='major', linewidth = 0.5, linestyle='-')
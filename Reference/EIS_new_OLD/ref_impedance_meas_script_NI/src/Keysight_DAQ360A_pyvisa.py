# !/usr/bin/env python3
#
# Digistant4463 Communication Pyvisa Serial Library.
#
# 0.0.1: Initial version

import pyvisa
import time
import numpy as np
import pandas as pd

""" Module to handle communication Digistant4463  """

__version__ = "0.0.7"  # semVersion (Major.Minor.Revision)

class Keysight_DAQ370A:

        def __init__(self, adress):

                rm = pyvisa.ResourceManager()
                time.sleep(0.1)

                self.adress = adress
        
                self.instrument = rm.open_resource(adress)
                time.sleep(0.1)

        def __str__(self):
                return f'Manual: General Instruction - page 8 - Queries and Commands, for details print object.__doc__'

        def identification(self, message="*IDN?"):
                return self.instrument.query(message)

        def remote(self, message="SYST:REM"):
                return self.instrument.write(message)

        def reset(self, message="*RST"):
                return self.instrument.write(message)
                
        def turn_off(self, message="OUTP:STAT OFF"):
                return self.instrument.write(message)
                
        def set_output_voltage(self, voltage):
                message = 'SOUR:VDC:AMPL ' + str(voltage)
                return self.instrument.write(message)

        def set_timeout(self, timeout):
                self.instrument.timeout = timeout

        def get_cal_date(self, message='SYST:ACAL? DMM'):
                return self.instrument.write(message)

        def get_cal_temp(self, message='SYST:ACAL:TEMP? DMM'):
                return self.instrument.write(message)

        def set_internal_DMM(self, value):
                if value:
                        return self.instrument.write('INST:DMM ON')
                else:
                        return self.instrument.write('INST:DMM OFF')  

        def set_config_temp_measure_FRTD(self, value):
                if value:
                        return self.instrument.write('INST:DMM ON')
                else:
                        return self.instrument.write('INST:DMM OFF') 

        def set_all_relays_open(self):

                #self.instrument.write('INST:DMM OFF')
                self.instrument.write('ROUT:OPEN (@301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,399)')

        def set_relays_close(self, relay_list):

                #self.instrument.write('INST:DMM OFF')
                self.instrument.write('ROUT:CLOSE (@'+','.join(str(relay) for relay in relay_list)+')')

        def set_relays_open(self, relay_list):

                #self.instrument.write('INST:DMM OFF')
                self.instrument.write('ROUT:OPEN (@'+','.join(str(relay) for relay in relay_list)+')')
   
        # * MACROS

        def initialize(self):

                print(self.instrument.query('*IDN?'))
                self.instrument.timeout = 60000
                self.instrument.write('*RST')
                mux_cal = self.instrument.query('SYST:ACAL? DMM')
                mux_cal_temp = self.instrument.query('SYST:ACAL:TEMP? DMM')

                print('DAQ970A after calibration on', time.asctime(), 'temperature is', mux_cal_temp, 'C.')

                self.set_all_relays_open()


        def set_temp_meas_settings(self):

                self.instrument.write('INST:DMM ON')
                time.sleep(0.5)
                self.instrument.write('CONF:TEMP:FRTD 100,(@309)')
                self.instrument.write('TEMP:ZERO:AUTO ON')
                self.instrument.write('TEMP:TRAN:FRTD:OCOM ON')
                self.instrument.write('TEMP:NPLC 100')

        def meas_temp(self):

                #DAQ970A CS-100 shunt temperature
                self.set_temp_meas_settings()
                temp_shunt = float(self.instrument.query('READ?'))     
                return temp_shunt

        def set_voltage_meas_relays(self):
                self.set_all_relays_open()
                self.set_relays_close(relay_list = [399])
                self.set_relays_close(relay_list = [301, 318])

        # * MACROS

        def set_channel_acquire3(self, channel, range='MAX', type='DIFF', coupling='AC', mode='TIME', samp_count='DEF', samp_rate=100000):
                string = 'ACQ3:VOLT '+str(range)+','+str(type)+','+str(coupling)+','+str(mode)+','+str(samp_count)+','+str(samp_rate)+',(@'+str(channel)+')'
                self.instrument.write(string)
                
        def init3(self, channel_1, channel_2):
                self.instrument.write('INIT3 (@'+str(channel_1)+','+str(channel_2)+')')

        def fetch3(self, channel, sample_rate):
                string = self.instrument.query('FETCh3? (@'+str(channel)+')')

                measurement = string_to_np_array(string)
                time        = np.arange(len(measurement))/sample_rate

                df = pd.DataFrame({
                        'Voltage': measurement,
                        'Time': time
                        })

                return df

# Helper funtions
def string_to_np_array(data: str):
        """
        Converts a comma-separated string of floats into a NumPy array.
        
        Args:
                data (str): Input string containing comma-separated float values.
        
        Returns:
                np.ndarray: NumPy array of floats.
        """
        try:
                # Split the string by commas
                float_list = [float(x) for x in data.split(',')]
                return np.array(float_list)
        except ValueError as e:
                print(f"Error converting to float: {e}")
                return np.array([])


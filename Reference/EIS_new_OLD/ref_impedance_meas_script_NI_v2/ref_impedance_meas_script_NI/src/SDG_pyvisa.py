# !/usr/bin/env python3
#
# SDG Waveform generator Communication Pyvisa Serial Library.

import pyvisa
import time

""" Module to handle communication SDG waveform generator  """

__version__ = "0.1"  # semVersion (Major.Minor)

class SDG:

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
        
        def turn_on(self, channel):
                return self.instrument.write(f'{channel}:OUTP ON')
        
        def turn_off(self, channel):
                return self.instrument.write(f'{channel}:OUTP OFF')
        
        def set_waveform(self, channel, waveform):
                return self.instrument.write(f'{channel}:BSWV WVTP,{waveform}')

        def set_frequency(self, channel, frequency):
                return self.instrument.write(f'{channel}:BSWV FRQ,{frequency}')
        
        def set_amplitude_p2p(self, channel, amplitude):
                return self.instrument.write(f'{channel}:BSWV AMP,{amplitude}')

        def initialize(self, timeout, channel, frequency, voltage_amp_p2p):
                self.instrument.timeout = timeout
                self.identification()
                self.turn_off(channel = channel)
                self.set_waveform(channel = channel, waveform='sine')
                self.set_frequency(channel = channel, frequency=frequency)
                self.set_amplitude_p2p(channel = channel, amplitude = voltage_amp_p2p)
                self.turn_on(channel = channel)
                        

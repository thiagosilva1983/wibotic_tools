#!/usr/bin/env python3
import traceback
from typing import Dict, List, Tuple
import pathlib
import time
from enum import Enum
from contextlib import suppress
from io import StringIO
import math
import pickle
import argparse
import warnings
import re
from datetime import datetime
import pytz

import tzlocal

import boto3
import boto3.dynamodb
from boto3.dynamodb.conditions import Attr, Key
import boto3.dynamodb.table
from botocore.client import Config
from botocore.exceptions import ClientError

import pandas

import numpy as np

from matplotlib import rc as mpl_rc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import _keys

'''
Utility to view the Box Build history of an TR or OC and generate a report of power tests that were run.

Units can be looked up by Serial Number or MAC Address.

Reports can be batched by using the --loadfile <filename> option to load a text file containing Serial Numbers or
Mac Addresses, one per line.
'''

# Suppress specific warning
warnings.filterwarnings("ignore"), 

# Pre-defined database names from Box Build
class DatabaseName(Enum):
    """Collection of Box Build DB Table Names"""
    PRODUCTION = "HardwareProduction"
    DEVELOPMENT = "DevHardwareProduction"

class InputType(Enum):
    SERIALNUMBER = "SN: "
    MACADDRESS = "Mac: "
    UNKNOWN = "Unknown"

class ReportType(Enum):
    SELECT_FROM_DATA = 1 # Could be a OC or TR serial number or mac, use to data in the selected record to decide
    OC_REPORT = 2        # This mac came from the oc_mac field, force a OC report
    
# Helper function to determine if the user inputs is a serial number or mac address.
# If mac address, any ".", "-" or ":" are stripped and the bare string is returned.
# Also returns an Enum with the type detected
# Assumes any strings you send in are .upper()
def detect_serial_or_mac(user_input: str) -> Tuple[str, InputType]:
    # Determine if the input is a serial number (see serial number wiki entry for hos this was constructed)
    if re.match(r'^[0-9A-F]{10}$', user_input):
        return user_input, InputType.SERIALNUMBER
    elif re.match(r'^(WX[0-9A-F]{8}|A[0-9A-F]{9}|B[0-9A-F]{9})$', user_input):
        return user_input, InputType.SERIALNUMBER
    # Determine if the input is a MAC address
    elif re.match(r'^34D954[0-9A-F]{6}$', user_input):
        return user_input, InputType.MACADDRESS
    elif re.match(r'^34[:.\-]D9[:.\-]54[:.\-][0-9A-F]{2}[:.\-][0-9A-F]{2}[:.\-][0-9A-F]{2}$', user_input):
        mac_address = re.sub(r'[:.\-]', '', user_input)
        return mac_address, InputType.MACADDRESS
    else:
        return None, InputType.UNKNOWN
       
# Creates a .pdf using information from the selected box build record
def create_pdf_from_record(data: List[Dict], passed: bool, plot_pdf: PdfPages, selection) -> bool:

    def plot(
        axes: plt.Axes,
        device: str,
        value_title: str,
        label: str,
        linestyle: str,
        color: str,
    ):
        device_data = charge_test_data[device]
        with suppress(KeyError):
            axes.plot(
                device_data["Timestamp"],
                device_data[value_title],
                label=f"{device.upper()}: {label}",
                linestyle=linestyle,
                color=color,
            )

    # Slice out TR vs OC
    model_number = (data[selection]['config']['ids']['mn'])
    system_type = model_number[:2]
    if system_type != 'TR' and system_type != 'OC':
        print("This serial number is not for a TR or OC")
        return False
    
    # The datalog_ keys contain the log data from power tests, make of list of keys present
    datalog_names = [key for key in sorted(data[selection].keys()) if "datalog_" in key]
    if not datalog_names:
        print("The selected record does not contain any saved data values, unable to create the report")
        return False

    if not passed:
        try:
            response = input("The selected report is for a failed test, do you still want to generate a report? [y/N]").upper()
        except KeyboardInterrupt:
            print("")
            print("Exiting WiBotic Box Build Report Generator")
        
        if response != 'Y':
            return False
    
    # Translate result
    result = 'Passed' if passed else 'Failed'
        
    # TR Report
    if 'TR' in system_type and data[selection]["type"] == ReportType.SELECT_FROM_DATA:
        # TR Report generation
        fig, main_axs = plt.subplots()
        fig.set_figwidth(12)
        fig.set_figheight(8)
        sec_axs = main_axs.twinx()
        
        charge_test_data = {}
        charge_test_data["tr"] = pandas.read_csv(StringIO(data[selection]['datalog_wireless_charge']["tr"]))
        
        # Zero timestamps
        charge_test_data["tr"]["Timestamp"] -= charge_test_data["tr"]["Timestamp"].iloc[0]
        
        # Gather Messages (if any)
        if 'Messages' in charge_test_data['tr']:
            messages = charge_test_data['tr'][charge_test_data['tr']['Messages'].notnull()]
            collated_messages = [
                f"Time: {timestamp:03d} - {value}"
                for timestamp, value in zip(messages["Timestamp"], messages['Messages'])
            ]
        else:
            collated_messages = []
        
        # Calculate TR values
        charge_test_data["tr"]["CalcWPa"] = (charge_test_data["tr"]["VMonPa"] * charge_test_data["tr"]["IMonPa"])        
        
        # Create the plot
        plot(main_axs, "tr", "CalcWPa", "TX Power [W]", "solid", "r")
        plot(main_axs, "tr", "VMonPa", "VMonPa [V]", "solid", "k")
        plot(sec_axs, "tr", "TMonPa", "TMonPa [°C]", "dotted", "r")
        plot(sec_axs, "tr", "TMonAmb", "TMonAmb [°C]", "dotted", "#fdb147")

        sec_axs.axhline(
            data[selection].get("charge_test_ambient_temp", 0),
            label="Amb. Temp [°C]",
            color="g",
            linestyle="dotted",
        )

        main_axs.legend(loc="upper left")
        sec_axs.legend(loc="lower right", ncol=3)
        main_axs.set_title(f"Test: Wireless Charge | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}")
        main_axs.set_xlabel("Time [s]")
        main_axs.set_ylabel("Volts / Power")
        sec_axs.set_ylabel("Temp [C]")
        main_axs.grid(True)
        main_axs.set_ylim((0, int(math.ceil(main_axs.get_ybound()[1] / 50) * 50)))
        sec_axs.set_ylim((0, int(math.ceil(sec_axs.get_ybound()[1] / 67.5) * 67.5)))
        main_axs.set_yticks(np.linspace(0, main_axs.get_ybound()[1], 10))
        sec_axs.set_yticks(np.linspace(0, sec_axs.get_ybound()[1], 10))
        main_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:>4.0f}"))
        sec_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:<4.0f}"))
        if collated_messages:
            plt.figtext(
                x=0.05,
                y=0.05,
                s="\n".join(collated_messages),
                multialignment="left",
                verticalalignment="top",
            )

        # Put the Mac address on the plot
        plt.text(2,1, f"Mac: {data[selection]['mac']}")
            
        plt.savefig(plot_pdf, format="pdf", bbox_inches="tight")
        plt.close()
        
        # TR Also gets a table with the results of the manual tests and the temperature tolerance checker
        try:
            table_data = {'Test': ['ready_led', 'charging_led', 'fault_led', 'Fans', 'PA Temperature', 'DC-DC Tempersture'],
                            'Low Limit': ['NA', 'NA', 'NA', 'NA', f"{data[selection]['tolerance_checks']['Wireless Charging PA Temp [TMonPa]']['lower_limit']}", f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['lower_limit']}"],
                            'High Limit': ['NA', 'NA', 'NA', 'NA', f"{data[selection]['tolerance_checks']['Wireless Charging PA Temp [TMonPa]']['upper_limit']}", f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['upper_limit']}"],
                            'Actual': ['NA', 'NA', 'NA', 'NA', f"{data[selection]['tolerance_checks']['Wireless Charging PA Temp [TMonPa]']['actual']}", f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['actual']}"],
                            'Result': [f"{'Pass' if data[selection]['pass_fail_prompts']['ready_led'] else 'Fail'}",
                                    f"{'Pass' if data[selection]['pass_fail_prompts']['charging_led'] else 'Fail'}",
                                    f"{'Pass' if data[selection]['pass_fail_prompts']['fault_led'] else 'Fail'}",
                                    f"{'Pass' if data[selection]['pass_fail_prompts']['fan_1'] else 'Fail'}",
                                    f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging PA Temp [TMonPa]']['pass'] else 'Fail'}",
                                    f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['pass'] else 'Fail'}"]
                        }    
            df = pandas.DataFrame(table_data)        
            fig, ax = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(8)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            for (row, col), cell in table.get_celld().items():
                cell.set_height(0.09)
            ax.set_title(f"Test: Tolerance Checks | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}")        
            plt.savefig(plot_pdf, format="pdf")
            plt.close()
        except Exception as e:
            print(f"Tolerance check data is not available: {e=}")
    else:
        # OC Report generation - we can generate up to three reports, charge test, float test and wall power test. Some OC reports won't have data for all three.
        # Wireless charge
        if 'datalog_wireless_charge' in datalog_names:
            charge_test_data = {}
            charge_test_data['oc'] = pandas.read_csv(StringIO(data[selection]['datalog_wireless_charge']['oc']))
            charge_test_data['bat'] = pandas.read_csv(StringIO(data[selection]['datalog_wireless_charge']['bat']))
            
            # Zero timestamps
            charge_test_data['oc']["Timestamp"] -= charge_test_data['oc']["Timestamp"].iloc[0]   
            charge_test_data['bat']["Timestamp"] -= charge_test_data['bat']["Timestamp"].iloc[0]  
            
            # Calculate charge test values
            charge_test_data['oc']["CalcWBatt"] = charge_test_data['oc']["VMonBatt"] * charge_test_data['oc']["IBattery"]
            charge_test_data['oc']['Current'] = -charge_test_data['bat']['Current']
            charge_test_data['oc']['Power'] = -charge_test_data['bat']['Power']
            charge_test_data['bat']['Power'] = -charge_test_data['bat']['Current'] * charge_test_data['bat']['Voltage']
            
            # Plot
            fig, wireless_main_axs = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(8)
            wireless_sec_axs = wireless_main_axs.twinx()
            
            plot(wireless_main_axs, "oc", "CalcWBatt", "Batt Power [W]", "solid", "b")
            plot(wireless_main_axs, "bat", "Power", "Sim Power [W]", "solid", "g")
            plot(wireless_sec_axs, "oc", "TBoard", "TBoard [°C]", "dotted", "b")
            plot(wireless_sec_axs, "oc", "TCharger", "TCharger [°C]", "dotted", "m")
            plot(wireless_sec_axs, "oc", "VRect", "VRect  [V]", "dashed", "k")
            
            # Some OC reports are abbreviated and the data is a result of a system wireless test. In the case we don't know the actual model of the OC.
            if data[selection]["type"] == ReportType.SELECT_FROM_DATA:
                # We were given the serial number, so the model number should be available
                title = f"Test: Wireless Charge | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}"
            else:
                # We were given a Mac address, so the model will not be available
                title = f"Test: Wireless Charge | Mac: {data[selection]['oc_mac']} | Date/Time: {data[selection]['time']} | Result: {result}"

            wireless_main_axs.legend(loc="upper left")
            wireless_sec_axs.legend(loc="lower right", ncol=3)
            wireless_main_axs.set_title(title)
            wireless_main_axs.set_xlabel("Time [s]")
            wireless_main_axs.set_ylabel("Power")
            wireless_sec_axs.set_ylabel("Volts / Temp [C]")
            wireless_main_axs.grid(True)
            wireless_main_axs.set_ylim((0, int(math.ceil(wireless_main_axs.get_ybound()[1] / 50) * 50)))
            wireless_sec_axs.set_ylim((0, int(math.ceil(wireless_sec_axs.get_ybound()[1] / 67.5) * 67.5)))
            wireless_main_axs.set_yticks(np.linspace(0, wireless_main_axs.get_ybound()[1], 10))
            wireless_sec_axs.set_yticks(np.linspace(0, wireless_sec_axs.get_ybound()[1], 10))
            wireless_main_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:>4.0f}"))
            wireless_sec_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:<4.0f}"))
        
            plt.savefig(plot_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)
            
        # Wall Power charge plot
        if 'datalog_wall_power_charge' in datalog_names:
            charge_test_data = {}
            charge_test_data['oc'] = pandas.read_csv(StringIO(data[selection]['datalog_wall_power_charge']['oc']))
            charge_test_data['bat'] = pandas.read_csv(StringIO(data[selection]['datalog_wall_power_charge']['bat']))
            
            # Zero timestamps
            charge_test_data['oc']["Timestamp"] -= charge_test_data['oc']["Timestamp"].iloc[0]   
            charge_test_data['bat']["Timestamp"] -= charge_test_data['bat']["Timestamp"].iloc[0]  
            
            # Calculate charge test values
            charge_test_data['oc']["CalcWBatt"] = charge_test_data['oc']["VMonBatt"] * charge_test_data['oc']["IBattery"]
            charge_test_data['oc']['Current'] = -charge_test_data['bat']['Current']
            charge_test_data['oc']['Power'] = -charge_test_data['bat']['Power']
            
            # Plot
            fig, wall_main_axs = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(8)
            wall_sec_axs = wall_main_axs.twinx()
    
            plot(wall_sec_axs, "oc", "IBattery", "IBattery  [A]", "dotted", "r")
            plot(wall_sec_axs, "bat", "Current", "Current  [A]", "dashed", "y")
            plot(wall_main_axs, "oc", "CalcWBatt", "Batt Power [W]", "solid", "b")
            plot(wall_main_axs, "oc", "Power", "Sim Power [W]", "solid", "g")
            plot(wall_sec_axs, "oc", "TBoard", "TBoard [°C]", "dotted", "b")
            plot(wall_sec_axs, "oc", "TCharger", "TCharger [°C]", "dotted", "m")
            plot(wall_sec_axs, "oc", "VRect", "VRect  [V]", "dashed", "k")

            wall_main_axs.legend(loc="upper left")
            wall_sec_axs.legend(loc="lower right", ncol=3)
            wall_main_axs.set_title(f"Test: Wall Power | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}")
            wall_main_axs.set_xlabel("Time [s]")
            wall_main_axs.set_ylabel("Power")
            wall_sec_axs.set_ylabel("Amps / Volts / Temp[C]")
            wall_main_axs.grid(True)
            wall_main_axs.set_ylim((0, int(math.ceil(wall_main_axs.get_ybound()[1] / 50) * 50)))
            wall_sec_axs.set_ylim((0, int(math.ceil(wall_sec_axs.get_ybound()[1] / 67.5) * 67.5)))
            wall_main_axs.set_yticks(np.linspace(0, wall_main_axs.get_ybound()[1], 10))
            wall_sec_axs.set_yticks(np.linspace(0, wall_sec_axs.get_ybound()[1], 10))
            wall_main_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:>4.0f}"))
            wall_sec_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:<4.0f}"))
        
            plt.savefig(plot_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)
            
        # Float Voltage Plot
        if 'datalog_float_voltage_test' in datalog_names:
            charge_test_data = {}
            charge_test_data['oc'] = pandas.read_csv(StringIO(data[selection]['datalog_float_voltage_test']['oc']))
            charge_test_data['bat'] = pandas.read_csv(StringIO(data[selection]['datalog_float_voltage_test']['bat']))
            
            # Zero timestamps
            charge_test_data['oc']["Timestamp"] -= charge_test_data['oc']["Timestamp"].iloc[0]   
            charge_test_data['bat']["Timestamp"] -= charge_test_data['bat']["Timestamp"].iloc[0]  
            
            # Plot
            fig, float_main_axs = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(8)
    
            plot(float_main_axs, "oc", "VMonBatt", "VMonBat  [V]", "solid", "b")
            plot(float_main_axs, "bat", "Voltage", "Voltage  [V]", "solid", "g")

            float_main_axs.legend(loc="upper left")
            float_main_axs.set_title(f"Test: Float Voltage | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}")
            float_main_axs.set_xlabel("Time [s]")
            float_main_axs.set_ylabel("Voltage")
            float_main_axs.grid(True)
            float_main_axs.set_yticks(np.linspace(0, float_main_axs.get_ybound()[1], 10))
            float_main_axs.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f"{x:>4.0f}"))
        
            plt.savefig(plot_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)
            
        # OC Also gets a table with the results of the tolerance checker
        #
        # BUT! If the report_type is OC_REPORT (which means the mac address was found in the oc_mac field), the power test data
        # was generated while a TR power test was run - and this also means that limited tolerance checks were run.
        # This is how we caurrently generate OC-1000 reports, since there is not a dedicated OC-1000 Wireless Charge test yet.
        try:
            if data[selection]["type"] == ReportType.SELECT_FROM_DATA:
                # This is a full OC report
                table_data = {'Test': ['Median\n Volt vs. Sim', 'Median\n Float vs. Sim', 'Median\n Float vs. Setpoint', 'Charge Current', 'Median\n Current vs. Sim', 'Median Current\n vs. OC Max Setting', 'TCharger', 'TBoard', 'TDC-DC'],
                                'Low Limit': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Charger Voltage']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Setpoint']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs Bat Sim']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs OC Max Setting']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['lower_limit']}",],                        
                                'High Limit': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Charger Voltage']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Setpoint']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs Bat Sim']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs OC Max Setting']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['upper_limit']}",],                        
                                'Actual': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Charger Voltage']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Float Voltage vs Setpoint']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs Bat Sim']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Median: OC Charge Current vs OC Max Setting']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['actual']}",],                        
                                'Result': [f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Float Voltage vs Charger Voltage']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Float Voltage vs Setpoint']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Charge Current vs Bat Sim']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Charge Current vs OC Max Setting']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging DC-DC Temp [TMonAmb]']['pass'] else 'Fail'}",]
                            }
                title = f"Test: Tolerance Checks | Model: {data[selection]['config']['ids']['mn']} | ID: {data[selection]['serial']} | Date/Time: {data[selection]['time']} | Result: {result}"
            else: 
                # This will be a more limited OC Report
                table_data = {'Test': ['Median\n Volt vs. Sim', 'Charge Current', 'TCharger', 'TBoard'],
                                'Low Limit': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['lower_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['lower_limit']}",],
                                'High Limit': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['upper_limit']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['upper_limit']}",],
                                'Actual': [f"{data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['actual']}",
                                                f"{data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['actual']}",],
                                'Result': [f"{'Pass' if data[selection]['tolerance_checks']['Median: OC Charge Voltage vs Bat Sim']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging Current [IBattery]']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging OC Charger Temp [TCharger]']['pass'] else 'Fail'}",
                                                f"{'Pass' if data[selection]['tolerance_checks']['Wireless Charging OC Board Temp [TBoard]']['pass'] else 'Fail'}",]
                }
                title = f"Test: Tolerance Checks | Mac: {data[selection]['oc_mac']} | Date/Time: {data[selection]['time']} | Result: {result}"
            
            # Decorate the report    
            df = pandas.DataFrame(table_data)        
            fig, ax = plt.subplots()
            fig.set_figwidth(12)
            fig.set_figheight(8)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            for (row, col), cell in table.get_celld().items():
                cell.set_height(0.1)
            ax.set_title(title)        
            plt.savefig(plot_pdf, format="pdf")
            plt.close(fig)
        except Exception as e:
            print(f"Tolerance Check data is not available, this page will be skipped: {e=}")   
            
    return True     

# Top-level report generator function, holds the pdf open for creation of mutiple pages                        
def create_report(data: List[Dict], sn_or_mac: str, selection, parent_path):
    
    # Let's make sure that is report has good data
    try:
        # Access to see if this generates an exception
        model_number = (data[selection]['config']['ids']['mn'])
    except KeyError:
        # If the record does not contain a model number, then no power tests are included
        print("The selected record does not contain any saved data values, unable to create the report")
        return
    system_type = model_number[:2]
    if system_type != 'TR' and system_type != 'OC':
        print("This serial number is not for a TR or OC")
        return
    
    # The datalog_ keys contain the log data from power tests, make of list of keys present
    datalog_names = [key for key in sorted(data[selection].keys()) if "datalog_" in key]
    if not datalog_names:
        print("The selected record does not contain any saved data values, unable to create the report")
        return
    
    print("Starting Report Generation...")
    passed = data[selection]['passed']
    file_name = f"{sn_or_mac}_data{'' if passed else '_failed'}.pdf"
    with PdfPages(parent_path / file_name) as pp:
        status = create_pdf_from_record(data, passed, pp, selection)
        if status:
            print(f"Report saved to: {parent_path / file_name}")
        else:
            print("No report was generated")

# Helper function for accessing AWS DynamoDB    
def get_db_table(table_name: DatabaseName):
    table_str: str = table_name.value
    dynamodb = boto3.Session(
        aws_access_key_id=_keys.ACCESS_KEY,
        aws_secret_access_key=_keys.SECRET_ACCESS_KEY,
    ).resource(
        "dynamodb",
        region_name="us-west-2",
        config=Config(connect_timeout=4, retries={"mode": "standard"}),
    )
    table = dynamodb.Table(table_str)
    return table

# Finds the passed serial number or mac address in the database.
# Serial numbers are always found in the 'serial' field.
# Mac addresses can be found in either the 'mac' field, or the 'oc_mac' field. Mac addresses
# in the 'mac' field can be TR or OC Mac addresses, depending on the type of procedure that was run.
def get_item_list_from_serial_or_mac(db: DatabaseName, table, serial_or_mac: str, type: InputType) -> list:

    found_items = []
    prompt_for_sure_flag = False
    
    # The InputType tells us if we are looking for a serial number or mac address.
    if type == InputType.SERIALNUMBER:
        index = "serial-index"
        key_condition = "serial"
        report_type = ReportType.SELECT_FROM_DATA
        print("Query the serial-index")
    elif type == InputType.MACADDRESS:
        # There are two mac indexes, "tr-mac-index" and "oc-mac-index". Start with TR and
        # if no results are returned, try OC
        index = "mac-index"
        key_condition = "mac"
        report_type = ReportType.SELECT_FROM_DATA
        print("Query the mac-index")
    else:
        print("Unknown InputType? (should not end up here)")
        return None

    while True:    
        # Make the initial database query
        try:
            resp = table.query(TableName=db.value, IndexName=index, KeyConditionExpression=Key(key_condition).eq(serial_or_mac), Limit=100)
            tmp_items = resp["Items"]
            print_string = f"Found: {len(tmp_items)} matches "
            print("\r", end="", flush=True)
            print(print_string, end="", flush=True)
 
            # Annotate each entry with the report_type
            for entry in tmp_items:
                entry["type"] = report_type   
            
            # And save (if there are any to save)
            if len(tmp_items) > 0:                
                found_items.extend(tmp_items)

        except ClientError as e:
            print(f"Error communicating with the database: {e}")
            return None, None
        
        while 'LastEvaluatedKey' in resp:
            # The number of returned items don't fit in a single response, we need to ask for more until done
            resp = table.query(TableName=db.value, IndexName=index, KeyConditionExpression=Key(key_condition).eq(serial_or_mac), Limit=100, ExclusiveStartKey=resp['LastEvaluatedKey'])
            tmp_items = resp["Items"]
            print_string = f"Found: {len(tmp_items)} additional items, {len(found_items)} total items "
            print("\r", end="", flush=True)
            print(print_string, end="", flush=True)
            
            # Annotate each entry with the report_type
            for entry in tmp_items:
                entry["type"] = report_type
            
            # And save (if there are any to save)
            if len(tmp_items) > 0:                
                found_items.extend(tmp_items)
                
            # If the count is large, the serial number may be for a test or golden unit that is present in many box-build results.
            # Let's make sure that the user really wants to see all the entries for this unit
            if len(found_items) > 30:
                if not prompt_for_sure_flag:
                    prompt_for_sure_flag = True
                    print("")
                    print("--------------------------------------------------")
                    print("This unit is present in a large number of records.")
                    print("It may be because this is a test or golden unit.")
                    response = input("Are you sure you want to continue scanning the database? [y/N]: ").upper()
                    if response != "Y":
                        return None

        # If we were scanning the mac-index update paraneters to scan the 'oc_mac' field in case this was an OC used in other wireless power tests
        if index == "mac-index":
            index = "oc_mac-index"
            key_condition = "oc_mac"
            report_type = ReportType.OC_REPORT
            print("")
            print("Query the oc-mac-index")
        else:
            print("...complete")
            break
    return found_items

# Prompts for inputs and runs reports
def create_data_report(args, item_list):
    list_count = 0
    db_open = False

    # Create the data directory if it does not exist    
    parent_path = pathlib.Path("data_reports/")
    parent_path.mkdir(parents=True, exist_ok=True)  # Make if not existing
                
    while True:
        # If loadfile is specfied, assume we have rows of serial numbers or mac addresses to prcess
        if args.loadfile != None:
            if list_count < len(item_list):
                list_entry = item_list[list_count]
                print(f"File entry is: {list_entry}")
                sn_or_mac, input_type = detect_serial_or_mac(list_entry)
                list_count += 1
            else:
                print("End of file reached, exiting")
                return
        else:
            try:
                user_input = input("Enter a serial number or mac address: ").strip().upper()                    
                sn_or_mac, input_type = detect_serial_or_mac(user_input)
            except KeyboardInterrupt:
                print("Keyboard interrupt, exiting...")
                return

        if input_type == InputType.UNKNOWN:
            print("The entered value is not a valid serial number or mac address. Try again.")
            continue
        
        if args.loadpickle:
            # Bypass database and load a previously saved dataset. This is for testing/debugging
            with open(f"{parent_path}/{sn_or_mac}_data.pickle", "rb") as file:
                data_object = pickle.load(file)
        else:    
            # Go to the database
            if not db_open:
                db_selection = DatabaseName.PRODUCTION if not args.development else DatabaseName.DEVELOPMENT
                dynamodb_table = get_db_table(db_selection)
                data_object = get_item_list_from_serial_or_mac(db_selection, dynamodb_table, sn_or_mac, input_type)
                db_open = True
        
        # Create a summary so the user can select the report
        summary_list = []
        if data_object != None:
            count = len(data_object)
            for i in range(count):
                dt_utc = datetime.fromisoformat(data_object[i]['create_time'])
                dt_utc.replace(tzinfo=pytz.UTC)
                local_tz = tzlocal.get_localzone()
                dt_local = dt_utc.astimezone(local_tz)
                summary_list.append([dt_local.strftime("%Y-%m-%d %H:%M:%S"), data_object[i]['config']['procedure_name'], 'Passed' if data_object[i]['passed'] else 'Failed', data_object[i]['serial']])

            # Assumes that create time is the 1st element in the sub-list
            sorted_summary_list = sorted(summary_list, key=lambda x: x[0])
        else:
            count = 0
            
        # If multiple items, sort by time
        if count > 1:
            i = 1                                                           
            for list_item in sorted_summary_list:
                print(f"{i}: {list_item}")
                i += 1

        if count == 0:
            print("No records found with this serial number")
            return False
        elif count == 1:
            while True:
                try:
                    print(f"1: {sorted_summary_list[0]}")
                    yes_no = input("Generate report from this record [Y/n]: ").upper()
                except KeyboardInterrupt:
                    print("")
                    print("Exiting WiBotic Box Build Report Generator")
                    return
                if yes_no != 'N':
                    selection = 0
                    break
                return
        else:
            while True:
                try:
                    entry = input(f"Select record for report [1-{count}]: ")
                    if entry.isdigit():
                        selection = int(entry)
                    else:
                        print("Please enter a number")
                        continue
                except KeyboardInterrupt:
                    print("")
                    print("Exiting WiBotic Box Build Report Generator")
                    return
                
                if selection < 1 or selection > count:
                    print(f"You must enter a number between 1 and {count}")
                else:
                    selection -= 1    # List starts at zero
                    break
            # Convert the selection back to a array index
            for i in range(count):
                if summary_list[i][0] == sorted_summary_list[selection][0]:
                    selection = i
                    break
        
        # Save the dataset to disk for use later. This is for testing/debug
        if args.savepickle:    
            with open(parent_path / pathlib.Path(f"{sn_or_mac}_data.pickle"), "wb") as file:
                pickle.dump(data_object, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved database object to: {parent_path / pathlib.Path(f"{sn_or_mac}_data.pickle")}")
                return
            
        create_report(data_object, sn_or_mac, selection, parent_path)
            
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='WiBotic Box Build Report Generator')
        parser.add_argument('--savepickle', action='store_true', default=False, help='Pulls serial number data from the database and pickles the data to disk. (development only)')
        parser.add_argument('--loadpickle', action='store_true', default=False, help='Loads data from a saved pickle, rather than the database. (development only)')
        parser.add_argument('--development', action='store_true', default=False, help='Use the development, not production database (development only)')
        parser.add_argument("--loadfile", type=str, default=None, help="Name of the text file containing serial numbers or Mac addresses to process. The file must have one S/N or Mac per line. Each will be processed sequentially.")
        args = parser.parse_args()        
        
        if args.loadfile:
            with open(args.loadfile, mode='r', newline='') as file:
                snlist = [line.strip() for line in file] 
        else:
            snlist = None
            
        create_data_report(args, snlist)
        
    except KeyboardInterrupt:
        print("")
        print("Exiting WiBotic Box Build Report Generator")
        exit(0)
    except SystemExit:
        print("")
        print("Exiting WiBotic Box Build Report Generator on SystemExit")
        exit(0)
    except Exception as e:
        print("")
        traceback.print_exception(e)
        exit(1)
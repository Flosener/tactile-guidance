#! /usr/bin/env python
# encoding: utf-8
import logging
import sys

import serial
import pybelt
from serial.tools import list_ports

from pybelt.belt_controller import BeltController, BeltConnectionState
from pybelt.belt_scanner import BeltScanner


def setup_logger():
    """Setups the logger to output debug of pyBelt on console.
    """
    logger = pybelt.logger

    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh_format = logging.Formatter('\033[92m %(levelname)s: %(message)s \033[0m')
    sh.setFormatter(sh_format)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)


def interactive_belt_connect(belt_controller):
    """Interactive procedure to connect a belt. The interface to use is asked via the console.

    :param BeltController belt_controller: The belt controller to connect.
    """

    # Ask for the interface
    #interface = input("Connect via Bluetooth or USB? [b,u]")
    #interface = ""
    #print("Connect via Bluetooth or USB? [b,u]", end="")
    #while interface == "":
    interface = input()
    #interface = 'b'
    if interface.lower() == "b":
        # Scan for advertising belt
        with pybelt.belt_scanner.create() as scanner:
            print("Start BLE scan.")
            belts = scanner.scan()
            print("BLE scan completed.")
        if len(belts) == 0:
            print("No belt found.")
            return belt_controller
        if len(belts) > 1:
            print("Select the belt to connect.")
            for i, belt in enumerate(belts):
                print("{}. {} - {}".format((i + 1), belt.name, belt.address))
            belt_selection = input("[1-{}]".format(len(belts)))
            try:
                belt_selection_int = int(belt_selection)
            except ValueError:
                print("Unrecognized input.")
                return belt_controller
            print("Connect the belt.")
            belt_controller.connect(belts[belt_selection_int - 1])
        else:
            print("Connect the belt.")
            belt_controller.connect(belts[0])

    elif interface.lower() == "u":
        all_ports = serial.tools.list_ports.comports()

        # Keep only real USB serial devices (ttyACM*/ttyUSB*); drop the legacy ttyS* ports
        ports = [p for p in all_ports
                if p.vid is not None
                or p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))]

        if len(ports) == 0:
            print("No USB serial port found. Is the belt plugged in and switched on?")
            print("Detected ports:", [p.device for p in all_ports])
            return belt_controller

        if len(ports) == 1:
            print("Connect the belt on {} ({}).".format(ports[0].device, ports[0].description))
            belt_controller.connect(ports[0].device)
        else:
            print("Select the serial COM port to use.")
            for i, port in enumerate(ports):
                print("{}. {} - {}".format(i + 1, port.device, port.description))
            belt_selection = input("[1-{}]".format(len(ports)))
            try:
                belt_selection_int = int(belt_selection)
                port = ports[belt_selection_int - 1]
            except (ValueError, IndexError):
                print("Unrecognized input.")
                return belt_controller
            print("Connect the belt.")
            belt_controller.connect(port.device)

    else:
        print("Unrecognized input.")
        return belt_controller

    return belt_controller



def main():

    setup_logger()

    # Interactive script to connect the belt
    belt_controller = BeltController()
    interactive_belt_connect(belt_controller)
    if belt_controller.get_connection_state() == BeltConnectionState.CONNECTED:
        print("Connection successful.")
        print("Belt firmware version: {}".format(belt_controller.get_firmware_version()))
        print("Belt mode: {}".format(belt_controller.get_belt_mode()))
        print("Belt default vibration intensity: {}".format(belt_controller.get_default_intensity()))
        print("Disconnect belt.")
    else:
        print("Connection failed.")
    belt_controller.disconnect_belt()

    return 0


if __name__ == "__main__":
    main()

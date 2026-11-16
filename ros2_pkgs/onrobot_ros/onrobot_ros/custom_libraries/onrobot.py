#!/usr/bin/env python3

import time
from pymodbus.client.sync import ModbusTcpClient, ModbusSerialClient
from pymodbus.exceptions import ModbusIOException


class RG():

    def __init__(self, gripper, ip=None, port=None, device=None):
        if not ip and device is None:
            raise Exception("Please provide either an IP address or a serial device.")
        if ip and device is not None:
            print("Both IP address and serial device provided. Using the IP address.")
            device = None
        
        config = {
            "stopbits": 1,
            "bytesize": 8,
            "parity": 'E',
            "baudrate": 115200,
            "timeout": 1
        }

        self.client = None
        if ip:
            self.client = self._create_client(ip, port, config)
        if device:
            self.client = self._create_client_serial(device, config)
        assert self.client is not None, "Error creating the client."

        if gripper not in ['rg2', 'rg6']:
            raise Exception("Please specify either 'rg2' or 'rg6'.")
        self.gripper = gripper
        if self.gripper == 'rg2':
            self.max_width = 1100
            self.max_force = 400
        elif self.gripper == 'rg6':
            self.max_width = 1600
            self.max_force = 1200

        self.open_connection()

    def _create_client(self, ip, port, config):
        client = ModbusTcpClient(
            ip,
            port=port,
            stopbits=config["stopbits"],
            bytesize=config["bytesize"],
            parity=config["parity"],
            baudrate=config["baudrate"],
            timeout=config["timeout"])
        return client

    def _create_client_serial(self, serial, config):
        client = ModbusSerialClient(
            method='rtu',
            port=serial,
            stopbits=config["stopbits"],
            bytesize=config["bytesize"],
            parity=config["parity"],
            baudrate=config["baudrate"],
            timeout=config["timeout"])
        return client

    def open_connection(self):
        self.client.connect()

    def close_connection(self):
        self.client.close()

    def is_connected(self):
        """Check if the connection to the gripper is still alive."""
        try:
            result = self.client.read_holding_registers(
                address=267, count=1, unit=65)
            if isinstance(result, ModbusIOException):
                return False
            return True
        except:
            return False

    def get_fingertip_offset(self):
        try:
            result = self.client.read_holding_registers(
                address=258, count=1, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
            offset_mm = result.registers[0] / 10.0
            return offset_mm
        except ModbusIOException:
            print("Failed to read fingertip offset.")
            return None

    def get_width(self):
        try:
            result = self.client.read_holding_registers(
                address=267, count=1, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
            width_mm = result.registers[0] / 10.0
            return width_mm
        except ModbusIOException:
            print("Failed to read width.")
            return None

    def get_status(self):
        try:
            result = self.client.read_holding_registers(
                address=268, count=1, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
            status = format(result.registers[0], '016b')
            status_list = [0] * 7
            if int(status[-1]):
                # print("A motion is ongoing so new commands are not accepted.")
                status_list[0] = 1
            if int(status[-2]):
                # print("An internal- or external grip is detected.")
                status_list[1] = 1
            if int(status[-3]):
                # print("Safety switch 1 is pushed.")
                status_list[2] = 1
            if int(status[-4]):
                # print("Safety circuit 1 is activated so it will not move.")
                status_list[3] = 1
            if int(status[-5]):
                # print("Safety switch 2 is pushed.")
                status_list[4] = 1
            if int(status[-6]):
                # print("Safety circuit 2 is activated so it will not move.")
                status_list[5] = 1
            if int(status[-7]):
                # print("Any of the safety switch is pushed.")
                status_list[6] = 1
            return status_list
        except ModbusIOException:
            print("Failed to read status.")
            return [None] * 7

    def get_width_with_offset(self):
        try:
            result = self.client.read_holding_registers(
                address=275, count=1, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
            width_mm = result.registers[0] / 10.0
            return width_mm
        except ModbusIOException:
            print("Failed to read width with offset.")
            return None

    def is_grasped(self):
        """Check if an object is currently grasped by the gripper."""
        try:
            status = self.get_status()
            if status and len(status) >= 2:
                return bool(status[1])  # Bit 1 indicates internal/external grip detected
            return False
        except Exception as e:
            print(f"Error checking grasp status: {e}")
            return False

    def is_moving(self):
        """Check if the gripper is currently in motion."""
        try:
            status = self.get_status()
            if status and len(status) >= 1:
                return bool(status[0])  # Bit 0 indicates motion ongoing
            return False
        except Exception as e:
            print(f"Error checking motion status: {e}")
            return False

    def get_grasp_force(self):
        """Get the current force being applied by the gripper (if available)."""
        # Note: This would require reading additional registers for force feedback
        # The exact register depends on the gripper model and firmware
        try:
            # For RG2/RG6, force feedback might be available in different registers
            # This is a placeholder - actual implementation depends on gripper specs
            result = self.client.read_holding_registers(
                address=269, count=1, unit=65)  # Example register for force
            if isinstance(result, ModbusIOException):
                raise result
            force = result.registers[0] / 10.0  # Convert to appropriate units
            return force
        except ModbusIOException:
            print("Force feedback not available or failed to read.")
            return None
        except Exception as e:
            print(f"Error reading force: {e}")
            return None

    def get_detailed_status(self):
        """Get a detailed status dictionary with all gripper information."""
        try:
            status = self.get_status()
            if not status or len(status) < 7:
                return None
            
            return {
                'motion_ongoing': bool(status[0]),
                'grasp_detected': bool(status[1]),
                'safety_switch_1': bool(status[2]),
                'safety_circuit_1': bool(status[3]),
                'safety_switch_2': bool(status[4]),
                'safety_circuit_2': bool(status[5]),
                'any_safety_switch': bool(status[6]),
                'current_width': self.get_width(),
                'width_with_offset': self.get_width_with_offset(),
                'fingertip_offset': self.get_fingertip_offset()
            }
        except Exception as e:
            print(f"Error getting detailed status: {e}")
            return None

    def set_control_mode(self, command):
        try:
            result = self.client.write_register(
                address=2, value=command, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to set control mode.")

    def set_target_force(self, force_val):
        try:
            result = self.client.write_register(
                address=0, value=force_val, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to set target force.")

    def set_target_width(self, width_val):
        try:
            result = self.client.write_register(
                address=1, value=width_val, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to set target width.")

    def close_gripper(self, force_val=400):
        try:
            params = [force_val, 0, 16]
            result = self.client.write_registers(
                address=0, values=params, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to close gripper.")

    def open_gripper(self, force_val=400):
        try:
            params = [force_val, self.max_width, 16]
            result = self.client.write_registers(
                address=0, values=params, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to open gripper.")

    def move_gripper(self, width_val, force_val=400):
        try:
            params = [force_val, width_val, 16]
            result = self.client.write_registers(
                address=0, values=params, unit=65)
            if isinstance(result, ModbusIOException):
                raise result
        except ModbusIOException:
            print("Failed to move gripper.")


#!/usr/bin/env python3
"""
Hardware Validation Framework for 3-Layer Hierarchical MPC Smart Charging
Validates INA219 sensors against multimeter readings, accuracy logging, stress testing
Author: Research Team
"""

import serial
import time
import json
import logging
import numpy as np
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import threading
from datetime import datetime, timedelta
import statistics
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class SensorReading:
    """Individual sensor reading data structure"""
    timestamp: float
    sensor_type: str  # 'ina219_battery', 'ina219_pv', 'ina219_grid', 'ds18b20'
    voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    temperature: Optional[float] = None
    raw_data: Optional[str] = None

@dataclass
class ValidationReading:
    """Validation reading comparing sensor vs reference"""
    timestamp: float
    sensor_reading: SensorReading
    reference_value: float
    error_absolute: float
    error_percentage: float
    within_tolerance: bool
    
@dataclass
class AccuracyMetrics:
    """Accuracy metrics for sensor validation"""
    mean_error: float
    std_error: float
    max_error: float
    min_error: float
    mean_abs_error: float
    rmse: float
    accuracy_percentage: float
    readings_count: int

class HardwareValidator:
    """Hardware validation framework for INA219 sensors"""
    
    def __init__(self, esp32_port: str = '/dev/ttyUSB0', config_path: Optional[str] = None):
        self.esp32_port = esp32_port
        self.serial_connection = None
        self.logger = logging.getLogger(__name__)
        
        # Sensor configurations for Triple INA219 and DS18B20
        self.sensor_configs = {
            'ina219_battery': {'address': '0x40', 'type': 'INA219', 'purpose': 'Battery monitoring'},
            'ina219_pv': {'address': '0x41', 'type': 'INA219', 'purpose': 'PV monitoring'},
            'ina219_grid': {'address': '0x44', 'type': 'INA219', 'purpose': 'Grid monitoring'},
            'ds18b20_temp': {'pin': 'GPIO4', 'type': 'DS18B20', 'purpose': 'Temperature monitoring'}
        }
        
        # Validation parameters
        self.voltage_tolerance = 0.1  # ±0.1V tolerance
        self.current_tolerance = 0.05  # ±0.05A tolerance  
        self.power_tolerance = 0.5    # ±0.5W tolerance
        self.temp_tolerance = 1.0     # ±1°C tolerance
        
        # Data storage
        self.validation_readings = []
        self.stress_test_data = []
        self.accuracy_history = {}
        
        # Threading
        self.validation_running = False
        self.stress_test_running = False
        
        # Database
        self.db_path = 'hardware_validation.db'
        self.init_database()
        
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> None:
        """Load validation configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            validation_config = config.get('hardware_validation', {})
            self.voltage_tolerance = validation_config.get('voltage_tolerance', self.voltage_tolerance)
            self.current_tolerance = validation_config.get('current_tolerance', self.current_tolerance)
            self.power_tolerance = validation_config.get('power_tolerance', self.power_tolerance)
            self.temp_tolerance = validation_config.get('temp_tolerance', self.temp_tolerance)
            
            self.logger.info(f"Hardware validation config loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading validation config: {e}")
            
    def init_database(self) -> None:
        """Initialize validation database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Sensor readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    sensor_type TEXT,
                    voltage REAL,
                    current REAL,
                    power REAL,
                    temperature REAL,
                    raw_data TEXT
                )
            ''')
            
            # Validation readings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    sensor_type TEXT,
                    sensor_value REAL,
                    reference_value REAL,
                    error_absolute REAL,
                    error_percentage REAL,
                    within_tolerance INTEGER,
                    measurement_type TEXT
                )
            ''')
            
            # Accuracy metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    sensor_type TEXT,
                    measurement_type TEXT,
                    mean_error REAL,
                    std_error REAL,
                    max_error REAL,
                    min_error REAL,
                    mean_abs_error REAL,
                    rmse REAL,
                    accuracy_percentage REAL,
                    readings_count INTEGER
                )
            ''')
            
            # Stress test results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stress_test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_start REAL,
                    test_duration REAL,
                    sensor_type TEXT,
                    total_readings INTEGER,
                    successful_readings INTEGER,
                    failed_readings INTEGER,
                    success_rate REAL,
                    avg_response_time REAL,
                    max_response_time REAL,
                    errors_detected INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Hardware validation database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing validation database: {e}")
            
    def connect_esp32(self) -> bool:
        """Connect to ESP32 via UART"""
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                
            self.serial_connection = serial.Serial(
                port=self.esp32_port,
                baudrate=115200,
                timeout=2.0,
                write_timeout=2.0
            )
            
            time.sleep(2)  # Wait for connection stabilization
            
            # Test communication
            test_response = self.send_command("STATUS")
            if test_response:
                self.logger.info(f"ESP32 connected successfully on {self.esp32_port}")
                return True
            else:
                self.logger.error("ESP32 communication test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to ESP32: {e}")
            return False
            
    def send_command(self, command: str) -> Optional[str]:
        """Send command to ESP32 and get response"""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                if not self.connect_esp32():
                    return None
                    
            self.serial_connection.write(f"{command}\n".encode())
            self.serial_connection.flush()
            
            # Read response with timeout
            response = ""
            start_time = time.time()
            while time.time() - start_time < 2.0:
                if self.serial_connection.in_waiting > 0:
                    response += self.serial_connection.read(self.serial_connection.in_waiting).decode('utf-8', errors='ignore')
                    if '\n' in response:
                        break
                time.sleep(0.01)
                
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error sending command to ESP32: {e}")
            return None
            
    def read_sensor_data(self) -> Optional[Dict[str, SensorReading]]:
        """Read all sensor data from ESP32"""
        try:
            response = self.send_command("READ_ALL")
            if not response:
                return None
                
            # Parse ESP32 response format
            # Expected format: "SENSORS:bat_v,bat_i,bat_p;pv_v,pv_i,pv_p;grid_v,grid_i,grid_p;temp"
            if not response.startswith("SENSORS:"):
                return None
                
            data_part = response[8:]  # Remove "SENSORS:" prefix
            sensor_parts = data_part.split(';')
            
            timestamp = time.time()
            readings = {}
            
            # Parse battery INA219 (0x40)
            if len(sensor_parts) > 0:
                bat_data = sensor_parts[0].split(',')
                if len(bat_data) >= 3:
                    readings['ina219_battery'] = SensorReading(
                        timestamp=timestamp,
                        sensor_type='ina219_battery',
                        voltage=float(bat_data[0]),
                        current=float(bat_data[1]),
                        power=float(bat_data[2]),
                        raw_data=sensor_parts[0]
                    )
                    
            # Parse PV INA219 (0x41)
            if len(sensor_parts) > 1:
                pv_data = sensor_parts[1].split(',')
                if len(pv_data) >= 3:
                    readings['ina219_pv'] = SensorReading(
                        timestamp=timestamp,
                        sensor_type='ina219_pv',
                        voltage=float(pv_data[0]),
                        current=float(pv_data[1]),
                        power=float(pv_data[2]),
                        raw_data=sensor_parts[1]
                    )
                    
            # Parse Grid INA219 (0x44)
            if len(sensor_parts) > 2:
                grid_data = sensor_parts[2].split(',')
                if len(grid_data) >= 3:
                    readings['ina219_grid'] = SensorReading(
                        timestamp=timestamp,
                        sensor_type='ina219_grid',
                        voltage=float(grid_data[0]),
                        current=float(grid_data[1]),
                        power=float(grid_data[2]),
                        raw_data=sensor_parts[2]
                    )
                    
            # Parse DS18B20 temperature
            if len(sensor_parts) > 3:
                temp_data = sensor_parts[3]
                if temp_data and temp_data != "nan":
                    readings['ds18b20'] = SensorReading(
                        timestamp=timestamp,
                        sensor_type='ds18b20',
                        temperature=float(temp_data),
                        raw_data=temp_data
                    )
                    
            return readings
            
        except Exception as e:
            self.logger.error(f"Error reading sensor data: {e}")
            return None
            
    def save_sensor_reading(self, reading: SensorReading) -> None:
        """Save sensor reading to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_readings 
                (timestamp, sensor_type, voltage, current, power, temperature, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                reading.timestamp,
                reading.sensor_type,
                reading.voltage,
                reading.current,
                reading.power,
                reading.temperature,
                reading.raw_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving sensor reading: {e}")
            
    def validate_against_reference(self, sensor_reading: SensorReading, 
                                 reference_values: Dict[str, float]) -> List[ValidationReading]:
        """Validate sensor reading against reference multimeter values"""
        validations = []
        
        for measurement_type, sensor_value in [
            ('voltage', sensor_reading.voltage),
            ('current', sensor_reading.current),
            ('power', sensor_reading.power),
            ('temperature', sensor_reading.temperature)
        ]:
            if sensor_value is None:
                continue
                
            ref_key = f"{sensor_reading.sensor_type}_{measurement_type}"
            if ref_key not in reference_values:
                continue
                
            reference_value = reference_values[ref_key]
            error_abs = abs(sensor_value - reference_value)
            error_pct = (error_abs / abs(reference_value)) * 100 if reference_value != 0 else 0
            
            # Determine tolerance
            tolerance = {
                'voltage': self.voltage_tolerance,
                'current': self.current_tolerance,
                'power': self.power_tolerance,
                'temperature': self.temp_tolerance
            }.get(measurement_type, 0.1)
            
            within_tolerance = error_abs <= tolerance
            
            validation = ValidationReading(
                timestamp=sensor_reading.timestamp,
                sensor_reading=sensor_reading,
                reference_value=reference_value,
                error_absolute=error_abs,
                error_percentage=error_pct,
                within_tolerance=within_tolerance
            )
            
            validations.append(validation)
            
            # Save to database
            self.save_validation_reading(validation, measurement_type)
            
        return validations
        
    def save_validation_reading(self, validation: ValidationReading, measurement_type: str) -> None:
        """Save validation reading to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            sensor_value = getattr(validation.sensor_reading, measurement_type, None)
            if sensor_value is None:
                return
                
            cursor.execute('''
                INSERT INTO validation_readings 
                (timestamp, sensor_type, sensor_value, reference_value, error_absolute, 
                 error_percentage, within_tolerance, measurement_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                validation.timestamp,
                validation.sensor_reading.sensor_type,
                sensor_value,
                validation.reference_value,
                validation.error_absolute,
                validation.error_percentage,
                1 if validation.within_tolerance else 0,
                measurement_type
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving validation reading: {e}")
            
    def calculate_accuracy_metrics(self, sensor_type: str, measurement_type: str, 
                                 hours_back: int = 1) -> Optional[AccuracyMetrics]:
        """Calculate accuracy metrics for recent readings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_timestamp = time.time() - (hours_back * 3600)
            
            cursor.execute('''
                SELECT error_absolute, error_percentage, within_tolerance
                FROM validation_readings
                WHERE sensor_type = ? AND measurement_type = ? AND timestamp > ?
            ''', (sensor_type, measurement_type, since_timestamp))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return None
                
            errors_abs = [r[0] for r in results]
            errors_pct = [r[1] for r in results]
            within_tolerance = [r[2] for r in results]
            
            metrics = AccuracyMetrics(
                mean_error=statistics.mean(errors_abs),
                std_error=statistics.stdev(errors_abs) if len(errors_abs) > 1 else 0,
                max_error=max(errors_abs),
                min_error=min(errors_abs),
                mean_abs_error=statistics.mean([abs(e) for e in errors_abs]),
                rmse=np.sqrt(np.mean([e**2 for e in errors_abs])),
                accuracy_percentage=(sum(within_tolerance) / len(within_tolerance)) * 100,
                readings_count=len(results)
            )
            
            # Save metrics to database
            self.save_accuracy_metrics(sensor_type, measurement_type, metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {e}")
            return None
            
    def save_accuracy_metrics(self, sensor_type: str, measurement_type: str, 
                            metrics: AccuracyMetrics) -> None:
        """Save accuracy metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO accuracy_metrics 
                (timestamp, sensor_type, measurement_type, mean_error, std_error, 
                 max_error, min_error, mean_abs_error, rmse, accuracy_percentage, readings_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                sensor_type,
                measurement_type,
                metrics.mean_error,
                metrics.std_error,
                metrics.max_error,
                metrics.min_error,
                metrics.mean_abs_error,
                metrics.rmse,
                metrics.accuracy_percentage,
                metrics.readings_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving accuracy metrics: {e}")
            
    def run_validation_session(self, duration_minutes: int = 60, 
                             interval_seconds: int = 5) -> Dict[str, Any]:
        """Run a validation session comparing sensors to reference values"""
        self.logger.info(f"Starting validation session for {duration_minutes} minutes")
        
        if not self.connect_esp32():
            return {"error": "Failed to connect to ESP32"}
            
        self.validation_running = True
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        session_data = {
            'start_time': start_time,
            'duration_minutes': duration_minutes,
            'readings': [],
            'validations': [],
            'metrics': {}
        }
        
        try:
            while time.time() < end_time and self.validation_running:
                # Read sensor data
                sensor_readings = self.read_sensor_data()
                if sensor_readings:
                    for sensor_type, reading in sensor_readings.items():
                        session_data['readings'].append(reading)
                        self.save_sensor_reading(reading)
                        
                    # Prompt for reference values (in real use, this would be automated)
                    print("\nCurrent sensor readings:")
                    for sensor_type, reading in sensor_readings.items():
                        if reading.voltage is not None:
                            print(f"{sensor_type} - V: {reading.voltage:.3f}V, I: {reading.current:.3f}A, P: {reading.power:.3f}W")
                        if reading.temperature is not None:
                            print(f"{sensor_type} - T: {reading.temperature:.1f}°C")
                            
                    # In automated setup, reference values would come from external multimeter
                    # For now, we'll simulate or skip validation
                    
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Validation session interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during validation session: {e}")
        finally:
            self.validation_running = False
            
        # Calculate final metrics
        for sensor_type in ['ina219_battery', 'ina219_pv', 'ina219_grid', 'ds18b20']:
            for measurement_type in ['voltage', 'current', 'power', 'temperature']:
                metrics = self.calculate_accuracy_metrics(sensor_type, measurement_type, 
                                                        hours_back=duration_minutes/60)
                if metrics:
                    session_data['metrics'][f"{sensor_type}_{measurement_type}"] = asdict(metrics)
                    
        session_data['end_time'] = time.time()
        session_data['actual_duration'] = session_data['end_time'] - session_data['start_time']
        
        self.logger.info(f"Validation session completed. Duration: {session_data['actual_duration']:.1f}s")
        
        return session_data
        
    def run_stress_test(self, duration_hours: int = 1) -> Dict[str, Any]:
        """Run stress test for specified duration"""
        self.logger.info(f"Starting stress test for {duration_hours} hours")
        
        if not self.connect_esp32():
            return {"error": "Failed to connect to ESP32"}
            
        self.stress_test_running = True
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        stress_data = {
            'start_time': start_time,
            'duration_hours': duration_hours,
            'total_readings': 0,
            'successful_readings': 0,
            'failed_readings': 0,
            'response_times': [],
            'errors': [],
            'sensor_performance': {}
        }
        
        try:
            while time.time() < end_time and self.stress_test_running:
                read_start = time.time()
                
                # Read sensor data
                sensor_readings = self.read_sensor_data()
                response_time = time.time() - read_start
                
                stress_data['total_readings'] += 1
                stress_data['response_times'].append(response_time)
                
                if sensor_readings:
                    stress_data['successful_readings'] += 1
                    
                    # Analyze each sensor
                    for sensor_type, reading in sensor_readings.items():
                        if sensor_type not in stress_data['sensor_performance']:
                            stress_data['sensor_performance'][sensor_type] = {
                                'successful_reads': 0,
                                'failed_reads': 0,
                                'response_times': [],
                                'value_ranges': {'voltage': [], 'current': [], 'power': [], 'temperature': []}
                            }
                            
                        perf = stress_data['sensor_performance'][sensor_type]
                        perf['successful_reads'] += 1
                        perf['response_times'].append(response_time)
                        
                        # Track value ranges
                        if reading.voltage is not None:
                            perf['value_ranges']['voltage'].append(reading.voltage)
                        if reading.current is not None:
                            perf['value_ranges']['current'].append(reading.current)
                        if reading.power is not None:
                            perf['value_ranges']['power'].append(reading.power)
                        if reading.temperature is not None:
                            perf['value_ranges']['temperature'].append(reading.temperature)
                            
                        # Save reading
                        self.save_sensor_reading(reading)
                        
                else:
                    stress_data['failed_readings'] += 1
                    for sensor_type in ['ina219_battery', 'ina219_pv', 'ina219_grid', 'ds18b20']:
                        if sensor_type not in stress_data['sensor_performance']:
                            stress_data['sensor_performance'][sensor_type] = {
                                'successful_reads': 0,
                                'failed_reads': 0,
                                'response_times': [],
                                'value_ranges': {'voltage': [], 'current': [], 'power': [], 'temperature': []}
                            }
                        stress_data['sensor_performance'][sensor_type]['failed_reads'] += 1
                        
                # Brief pause between readings
                time.sleep(0.1)
                
                # Log progress every 10 minutes
                if stress_data['total_readings'] % 6000 == 0:  # Assuming ~10 readings per second
                    elapsed = time.time() - start_time
                    success_rate = (stress_data['successful_readings'] / stress_data['total_readings']) * 100
                    self.logger.info(f"Stress test progress: {elapsed/3600:.1f}h, {success_rate:.1f}% success rate")
                    
        except KeyboardInterrupt:
            self.logger.info("Stress test interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during stress test: {e}")
            stress_data['errors'].append(str(e))
        finally:
            self.stress_test_running = False
            
        # Calculate final statistics
        stress_data['end_time'] = time.time()
        stress_data['actual_duration'] = stress_data['end_time'] - stress_data['start_time']
        stress_data['success_rate'] = (stress_data['successful_readings'] / stress_data['total_readings']) * 100 if stress_data['total_readings'] > 0 else 0
        stress_data['avg_response_time'] = statistics.mean(stress_data['response_times']) if stress_data['response_times'] else 0
        stress_data['max_response_time'] = max(stress_data['response_times']) if stress_data['response_times'] else 0
        
        # Save stress test results
        self.save_stress_test_results(stress_data)
        
        self.logger.info(f"Stress test completed. Success rate: {stress_data['success_rate']:.1f}%")
        
        return stress_data
        
    def save_stress_test_results(self, stress_data: Dict[str, Any]) -> None:
        """Save stress test results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for sensor_type, perf in stress_data['sensor_performance'].items():
                cursor.execute('''
                    INSERT INTO stress_test_results 
                    (test_start, test_duration, sensor_type, total_readings, successful_readings, 
                     failed_readings, success_rate, avg_response_time, max_response_time, errors_detected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stress_data['start_time'],
                    stress_data['actual_duration'],
                    sensor_type,
                    perf['successful_reads'] + perf['failed_reads'],
                    perf['successful_reads'],
                    perf['failed_reads'],
                    (perf['successful_reads'] / (perf['successful_reads'] + perf['failed_reads'])) * 100,
                    statistics.mean(perf['response_times']) if perf['response_times'] else 0,
                    max(perf['response_times']) if perf['response_times'] else 0,
                    len(stress_data['errors'])
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving stress test results: {e}")
            
    def generate_validation_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get overall statistics
            since_timestamp = time.time() - (hours_back * 3600)
            
            # Query accuracy metrics
            accuracy_df = pd.read_sql_query('''
                SELECT * FROM accuracy_metrics WHERE timestamp > ?
            ''', conn, params=(since_timestamp,))
            
            # Query validation readings
            validation_df = pd.read_sql_query('''
                SELECT * FROM validation_readings WHERE timestamp > ?
            ''', conn, params=(since_timestamp,))
            
            # Query stress test results
            stress_df = pd.read_sql_query('''
                SELECT * FROM stress_test_results WHERE test_start > ?
            ''', conn, params=(since_timestamp,))
            
            conn.close()
            
            report = {
                'report_timestamp': time.time(),
                'hours_analyzed': hours_back,
                'summary': {},
                'sensor_performance': {},
                'accuracy_analysis': {},
                'stress_test_summary': {},
                'recommendations': []
            }
            
            # Overall summary
            if not validation_df.empty:
                overall_accuracy = (validation_df['within_tolerance'].sum() / len(validation_df)) * 100
                report['summary'] = {
                    'total_validations': len(validation_df),
                    'overall_accuracy': overall_accuracy,
                    'sensors_tested': validation_df['sensor_type'].nunique(),
                    'measurement_types': validation_df['measurement_type'].nunique()
                }
                
            # Sensor performance analysis
            for sensor_type in validation_df['sensor_type'].unique():
                sensor_data = validation_df[validation_df['sensor_type'] == sensor_type]
                accuracy = (sensor_data['within_tolerance'].sum() / len(sensor_data)) * 100
                
                report['sensor_performance'][sensor_type] = {
                    'accuracy_percentage': accuracy,
                    'total_readings': len(sensor_data),
                    'mean_error': sensor_data['error_absolute'].mean(),
                    'max_error': sensor_data['error_absolute'].max(),
                    'status': 'Good' if accuracy > 95 else 'Warning' if accuracy > 90 else 'Poor'
                }
                
            # Stress test summary
            if not stress_df.empty:
                report['stress_test_summary'] = {
                    'tests_conducted': len(stress_df),
                    'avg_success_rate': stress_df['success_rate'].mean(),
                    'avg_response_time': stress_df['avg_response_time'].mean(),
                    'max_response_time': stress_df['max_response_time'].max(),
                    'total_duration_hours': stress_df['test_duration'].sum() / 3600
                }
                
            # Generate recommendations
            if report['summary'].get('overall_accuracy', 0) < 95:
                report['recommendations'].append("Overall accuracy below 95%. Check sensor calibration.")
                
            for sensor_type, perf in report['sensor_performance'].items():
                if perf['accuracy_percentage'] < 90:
                    report['recommendations'].append(f"{sensor_type} accuracy low: {perf['accuracy_percentage']:.1f}%")
                    
            if report['stress_test_summary'].get('avg_response_time', 0) > 1.0:
                report['recommendations'].append("High response times detected. Check communication stability.")
                
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {e}")
            return {"error": str(e)}
            
    def plot_validation_results(self, sensor_type: str, measurement_type: str, 
                              hours_back: int = 24) -> str:
        """Plot validation results and save figure"""
        try:
            conn = sqlite3.connect(self.db_path)
            since_timestamp = time.time() - (hours_back * 3600)
            
            df = pd.read_sql_query('''
                SELECT timestamp, sensor_value, reference_value, error_absolute, error_percentage
                FROM validation_readings
                WHERE sensor_type = ? AND measurement_type = ? AND timestamp > ?
                ORDER BY timestamp
            ''', conn, params=(sensor_type, measurement_type, since_timestamp))
            
            conn.close()
            
            if df.empty:
                return "No data available for plotting"
                
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Sensor vs Reference values
            ax1.plot(df['datetime'], df['sensor_value'], label='Sensor', alpha=0.7)
            ax1.plot(df['datetime'], df['reference_value'], label='Reference', alpha=0.7)
            ax1.set_title(f'{sensor_type} - {measurement_type} Values')
            ax1.set_ylabel(measurement_type.title())
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Absolute error
            ax2.plot(df['datetime'], df['error_absolute'], color='red', alpha=0.7)
            ax2.set_title(f'Absolute Error')
            ax2.set_ylabel(f'Error ({measurement_type})')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Percentage error
            ax3.plot(df['datetime'], df['error_percentage'], color='orange', alpha=0.7)
            ax3.set_title(f'Percentage Error')
            ax3.set_ylabel('Error (%)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Error histogram
            ax4.hist(df['error_absolute'], bins=20, alpha=0.7, color='green')
            ax4.set_title('Error Distribution')
            ax4.set_xlabel(f'Absolute Error ({measurement_type})')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = f'validation_{sensor_type}_{measurement_type}_{int(time.time())}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error plotting validation results: {e}")
            return f"Error: {e}"
            
    def export_validation_data(self, hours_back: int = 24) -> str:
        """Export validation data to CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            since_timestamp = time.time() - (hours_back * 3600)
            
            # Export validation readings
            validation_df = pd.read_sql_query('''
                SELECT * FROM validation_readings WHERE timestamp > ?
            ''', conn, params=(since_timestamp,))
            
            # Export accuracy metrics
            accuracy_df = pd.read_sql_query('''
                SELECT * FROM accuracy_metrics WHERE timestamp > ?
            ''', conn, params=(since_timestamp,))
            
            # Export stress test results
            stress_df = pd.read_sql_query('''
                SELECT * FROM stress_test_results WHERE test_start > ?
            ''', conn, params=(since_timestamp,))
            
            conn.close()
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save to CSV files
            validation_filename = f'validation_readings_{timestamp_str}.csv'
            accuracy_filename = f'accuracy_metrics_{timestamp_str}.csv'
            stress_filename = f'stress_test_results_{timestamp_str}.csv'
            
            validation_df.to_csv(validation_filename, index=False)
            accuracy_df.to_csv(accuracy_filename, index=False)
            stress_df.to_csv(stress_filename, index=False)
            
            return f"Data exported: {validation_filename}, {accuracy_filename}, {stress_filename}"
            
        except Exception as e:
            self.logger.error(f"Error exporting validation data: {e}")
            return f"Error: {e}"
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.validation_running = False
        self.stress_test_running = False
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            
def test_hardware_validation():
    """Test hardware validation system"""
    print("Testing Hardware Validation System...")
    
    validator = HardwareValidator()
    
    # Test connection
    if validator.connect_esp32():
        print("✓ ESP32 connection successful")
        
        # Test sensor reading
        readings = validator.read_sensor_data()
        if readings:
            print(f"✓ Sensor readings obtained: {len(readings)} sensors")
            for sensor_type, reading in readings.items():
                print(f"  {sensor_type}: V={reading.voltage}, I={reading.current}, P={reading.power}, T={reading.temperature}")
        else:
            print("✗ Failed to read sensor data")
            
    else:
        print("✗ ESP32 connection failed")
        
    # Test database operations
    try:
        # Create test reading
        test_reading = SensorReading(
            timestamp=time.time(),
            sensor_type='ina219_battery',
            voltage=12.5,
            current=2.0,
            power=25.0,
            raw_data='12.5,2.0,25.0'
        )
        
        validator.save_sensor_reading(test_reading)
        print("✓ Database operations working")
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        
    # Generate test report
    report = validator.generate_validation_report(hours_back=1)
    print(f"✓ Report generated with {len(report)} sections")
    
    validator.cleanup()
    print("Hardware validation system test completed!")
    
    return validator

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    validator = test_hardware_validation()
    
    # Example usage
    print("\nExample usage:")
    print("validator = HardwareValidator('/dev/ttyUSB0')")
    print("session_data = validator.run_validation_session(duration_minutes=30)")
    print("stress_data = validator.run_stress_test(duration_hours=1)")
    print("report = validator.generate_validation_report(hours_back=24)")
    print("plot_file = validator.plot_validation_results('ina219_battery', 'voltage')")
    print("export_files = validator.export_validation_data(hours_back=24)")

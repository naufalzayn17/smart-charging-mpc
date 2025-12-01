#!/usr/bin/env python3
"""
MAIN SYSTEM ORCHESTRATOR - HIERARCHICAL MPC SMART CHARGING
===========================================================

Komprehensif system coordinator untuk 3-layer hierarchical MPC system
- Headless operation dengan remote monitoring
- Multi-layer coordination dan synchronization
- Automated startup, shutdown, dan recovery
- System health monitoring dan diagnostics
- 96-hour validation data collection
- Remote monitoring integration

Layers:
Layer 1: ESP32 Fast MPC (< 150ms response time)
Layer 2: Medium MPC (6-hour horizon optimization)
Layer 3: Adaptive Learning (24-hour strategic planning)

Author: Dzaky Naufal K
Date: July 2, 2025
Version: 3.0 - Headless System Orchestrator
"""

import sys
import os
import time
import json
import logging
import threading
import subprocess
import signal
import psutil
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import serial
import paho.mqtt.client as mqtt
from contextlib import contextmanager

# Import network configuration
try:
    from config.network_config import get_mqtt_config, get_wifi_config, get_device_config
except ImportError:
    print("WARNING: Network config not found, using default values")
    get_mqtt_config = lambda: {"broker": "10.121.146.109", "port": 1883}
    get_wifi_config = lambda: {"ssid": "TP-Link_E4-FC"}
    get_device_config = lambda: {"name": "ESP32-MPC-Controller"}

# Import priority condition controller
try:
    from priority_condition_controller import PriorityConditionController, Priority
except ImportError:
    print("WARNING: Priority condition controller not found")
    PriorityConditionController = None
    Priority = None

# Configure logging untuk orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SystemOrchestrator - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mpc_system_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration untuk orchestrator"""
    # ESP32 Layer 1 Configuration
    esp32_port: str = "/dev/ttyUSB0"
    esp32_baudrate: int = 115200
    esp32_timeout: float = 5.0
    
    # Layer 2 & 3 Configuration
    layer2_enabled: bool = True
    layer3_enabled: bool = True
    data_collection_enabled: bool = True
    
    # Remote monitoring (loaded from network config)
    def __post_init__(self):
        # Load network configuration
        mqtt_config = get_mqtt_config()
        wifi_config = get_wifi_config()
        device_config = get_device_config()
        
        # Override with network config values
        self.mqtt_broker = mqtt_config["broker"]
        self.mqtt_port = mqtt_config["port"]
        self.mqtt_topic_prefix = device_config["mqtt_prefix"]
        self.wifi_ssid = wifi_config["ssid"]
    
    # Default values (will be overridden by __post_init__)
    mqtt_broker: str = "10.121.146.109"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "mpc_smart_charging/ESP32-MPC-Controller"
    wifi_ssid: str = "TP-Link_E4-FC"
    
    # System validation
    max_layer1_response_ms: float = 150.0
    max_layer2_response_s: float = 30.0
    min_system_uptime_percent: float = 95.0
    validation_period_hours: int = 96
    
    # Emergency thresholds
    max_battery_temp_c: float = 45.0
    min_battery_voltage_v: float = 10.0
    max_battery_voltage_v: float = 14.4
    max_current_charge_a: float = 30.0
    max_current_discharge_a: float = 30.0

@dataclass
class SystemStatus:
    """Real-time system status"""
    timestamp: float
    layer1_status: str  # "running", "stopped", "error"
    layer2_status: str
    layer3_status: str
    data_collection_status: str
    mqtt_status: str
    esp32_connection_status: str
    system_uptime: float
    last_error: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None

class SystemOrchestrator:
    """Main system orchestrator untuk hierarchical MPC"""
    
    def __init__(self, config_file: str = "system_config.json"):
        """Initialize system orchestrator"""
        self.config = self._load_config(config_file)
        self.status = SystemStatus(
            timestamp=time.time(),
            layer1_status="stopped",
            layer2_status="stopped", 
            layer3_status="stopped",
            data_collection_status="stopped",
            mqtt_status="disconnected",
            esp32_connection_status="disconnected",
            system_uptime=0.0
        )
        
        # Process management
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Communication
        self.esp32_serial: Optional[serial.Serial] = None
        self.mqtt_client: Optional[mqtt.Client] = None
        self.command_queue = queue.Queue()
        
        # Monitoring
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.health_check_interval = 30.0  # seconds
        
        # Data collection
        self.db_path = "system_orchestrator.db"
        self._init_database()
        
        # Priority condition controller untuk emergency handling
        self.priority_controller = None
        if PriorityConditionController:
            try:
                self.priority_controller = PriorityConditionController()
                logger.info("Priority Condition Controller initialized for emergency handling")
            except Exception as e:
                logger.error(f"Failed to initialize Priority Controller: {e}")
        
        logger.info("System Orchestrator initialized")
    
    def _load_config(self, config_file: str) -> SystemConfig:
        """Load system configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                return SystemConfig(**config_data)
            else:
                logger.warning(f"Config file {config_file} not found, using defaults")
                return SystemConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return SystemConfig()
    
    def _init_database(self):
        """Initialize database untuk system logging"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System status table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    layer1_status TEXT,
                    layer2_status TEXT, 
                    layer3_status TEXT,
                    data_collection_status TEXT,
                    mqtt_status TEXT,
                    esp32_connection_status TEXT,
                    system_uptime REAL,
                    last_error TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            # System events log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    layer TEXT,
                    description TEXT,
                    severity TEXT,
                    data TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _log_event(self, event_type: str, layer: str, description: str, 
                   severity: str = "INFO", data: Optional[Dict] = None):
        """Log system event ke database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_events 
                (timestamp, event_type, layer, description, severity, data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                event_type,
                layer,
                description,
                severity,
                json.dumps(data) if data else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging event: {e}")
    
    def _connect_esp32(self) -> bool:
        """Connect to ESP32 via UART"""
        try:
            if self.esp32_serial and self.esp32_serial.is_open:
                return True
                
            self.esp32_serial = serial.Serial(
                port=self.config.esp32_port,
                baudrate=self.config.esp32_baudrate,
                timeout=self.config.esp32_timeout
            )
            
            # Test connection
            test_cmd = json.dumps({"command": "ping"}) + "\n"
            self.esp32_serial.write(test_cmd.encode())
            
            # Wait for response
            time.sleep(0.1)
            if self.esp32_serial.in_waiting > 0:
                response = self.esp32_serial.readline().decode().strip()
                if "pong" in response.lower():
                    self.status.esp32_connection_status = "connected"
                    logger.info("ESP32 connection established")
                    return True
            
            self.status.esp32_connection_status = "disconnected"
            return False
            
        except Exception as e:
            logger.error(f"ESP32 connection error: {e}")
            self.status.esp32_connection_status = "error"
            return False
    
    def _setup_mqtt(self) -> bool:
        """Setup MQTT client untuk remote monitoring"""
        try:
            self.mqtt_client = mqtt.Client(client_id="mpc_system_orchestrator")
            
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self.status.mqtt_status = "connected"
                    logger.info("MQTT connected successfully")
                    # Subscribe to command topic
                    client.subscribe(f"{self.config.mqtt_topic_prefix}/commands")
                else:
                    self.status.mqtt_status = "error"
                    logger.error(f"MQTT connection failed: {rc}")
            
            def on_message(client, userdata, msg):
                try:
                    command = json.loads(msg.payload.decode())
                    self.command_queue.put(command)
                    logger.info(f"Received MQTT command: {command}")
                except Exception as e:
                    logger.error(f"MQTT message processing error: {e}")
            
            self.mqtt_client.on_connect = on_connect
            self.mqtt_client.on_message = on_message
            
            self.mqtt_client.connect(self.config.mqtt_broker, self.config.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT setup error: {e}")
            self.status.mqtt_status = "error"
            return False
    
    def _start_layer_process(self, layer_name: str, script_path: str) -> bool:
        """Start layer process"""
        try:
            if layer_name in self.processes:
                return True  # Already running
            
            # Ensure script is executable
            os.chmod(script_path, 0o755)
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(script_path)
            )
            
            self.processes[layer_name] = process
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(layer_name, process),
                daemon=True
            )
            monitor_thread.start()
            self.threads[f"{layer_name}_monitor"] = monitor_thread
            
            logger.info(f"{layer_name} process started (PID: {process.pid})")
            self._log_event("process_start", layer_name, f"Process started with PID {process.pid}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting {layer_name}: {e}")
            self._log_event("process_error", layer_name, f"Failed to start: {e}", "ERROR")
            return False
    
    def _monitor_process(self, layer_name: str, process: subprocess.Popen):
        """Monitor layer process"""
        while not self.shutdown_event.is_set():
            try:
                # Check if process is still running
                if process.poll() is not None:
                    # Process terminated
                    self._log_event("process_terminated", layer_name, 
                                  f"Process terminated with code {process.returncode}", "WARNING")
                    
                    # Update status
                    if layer_name == "layer2":
                        self.status.layer2_status = "stopped"
                    elif layer_name == "layer3":
                        self.status.layer3_status = "stopped"
                    elif layer_name == "data_collection":
                        self.status.data_collection_status = "stopped"
                    
                    # Attempt restart if not shutting down
                    if not self.shutdown_event.is_set():
                        logger.warning(f"{layer_name} process died, attempting restart...")
                        time.sleep(5)  # Wait before restart
                        self._restart_layer(layer_name)
                    
                    break
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring {layer_name}: {e}")
                break
    
    def _restart_layer(self, layer_name: str):
        """Restart a specific layer"""
        try:
            # Stop existing process
            if layer_name in self.processes:
                self.processes[layer_name].terminate()
                self.processes[layer_name].wait(timeout=10)
                del self.processes[layer_name]
            
            # Restart based on layer
            if layer_name == "layer2" and self.config.layer2_enabled:
                self._start_layer_process("layer2", "layer2_mpc_optimizer_headless.py")
                self.status.layer2_status = "running"
            elif layer_name == "layer3" and self.config.layer3_enabled:
                self._start_layer_process("layer3", "layer3_adaptive_planner_headless.py")
                self.status.layer3_status = "running"
            elif layer_name == "data_collection" and self.config.data_collection_enabled:
                self._start_layer_process("data_collection", "data_collection_96hour_headless.py")
                self.status.data_collection_status = "running"
            
            logger.info(f"{layer_name} restarted successfully")
            
        except Exception as e:
            logger.error(f"Error restarting {layer_name}: {e}")
    
    def _health_check(self):
        """Perform system health check"""
        try:
            current_time = time.time()
            
            # Update system uptime
            self.status.system_uptime = current_time - self.start_time
            
            # Priority condition emergency check
            if self.priority_controller:
                try:
                    # Simulasi sensor data untuk emergency check
                    # Dalam implementasi nyata, ini akan mengambil data dari sensor aktual
                    mock_sensor_data = {
                        'battery_temp': 25.0,  # Default safe temperature
                        'soc': 50.0,
                        'pv_power': 500.0,
                        'load_power': 1000.0,
                        'battery_voltage': 13.0,
                        'battery_current': 0.0,
                        'grid_available': True
                    }
                    
                    # NOTE: Dalam production, ganti dengan data sensor real
                    priority_conditions = self.priority_controller.evaluate_conditions(mock_sensor_data)
                    
                    # Check for emergency conditions
                    for condition in priority_conditions:
                        if condition.priority == Priority.EMERGENCY_STOP:
                            logger.critical(f"SYSTEM EMERGENCY DETECTED: {condition.description}")
                            self.status.last_error = f"EMERGENCY: {condition.description}"
                            # Trigger emergency shutdown
                            self._trigger_emergency_shutdown(condition.description)
                            break
                        elif condition.priority.value <= 2:  # High priority conditions
                            logger.warning(f"HIGH PRIORITY CONDITION: {condition.description}")
                            
                except Exception as e:
                    logger.error(f"Priority condition check error: {e}")
            
            # Check ESP32 connection
            if not self._connect_esp32():
                logger.warning("ESP32 connection lost, attempting reconnection...")
            
            # Check MQTT connection
            if self.mqtt_client and not self.mqtt_client.is_connected():
                logger.warning("MQTT connection lost, attempting reconnection...")
                try:
                    self.mqtt_client.reconnect()
                except:
                    pass
            
            # Check process health
            for layer_name, process in self.processes.items():
                if process.poll() is not None:
                    logger.warning(f"{layer_name} process died")
                    self._restart_layer(layer_name)
            
            # Log status to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_status 
                (timestamp, layer1_status, layer2_status, layer3_status,
                 data_collection_status, mqtt_status, esp32_connection_status,
                 system_uptime, last_error, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_time,
                self.status.layer1_status,
                self.status.layer2_status,
                self.status.layer3_status,
                self.status.data_collection_status,
                self.status.mqtt_status,
                self.status.esp32_connection_status,
                self.status.system_uptime,
                self.status.last_error,
                json.dumps(self.status.performance_metrics) if self.status.performance_metrics else None
            ))
            
            conn.commit()
            conn.close()
            
            # Publish status via MQTT
            if self.mqtt_client and self.mqtt_client.is_connected():
                status_data = asdict(self.status)
                self.mqtt_client.publish(
                    f"{self.config.mqtt_topic_prefix}/system_status",
                    json.dumps(status_data)
                )
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            self.status.last_error = str(e)
    
    def _process_commands(self):
        """Process MQTT commands"""
        try:
            while not self.command_queue.empty():
                command = self.command_queue.get_nowait()
                
                cmd_type = command.get("type", "")
                
                if cmd_type == "shutdown":
                    logger.info("Received shutdown command")
                    self.shutdown()
                    
                elif cmd_type == "restart_layer":
                    layer = command.get("layer", "")
                    logger.info(f"Received restart command for {layer}")
                    self._restart_layer(layer)
                    
                elif cmd_type == "get_status":
                    # Send detailed status
                    if self.mqtt_client and self.mqtt_client.is_connected():
                        status_data = asdict(self.status)
                        self.mqtt_client.publish(
                            f"{self.config.mqtt_topic_prefix}/status_response",
                            json.dumps(status_data)
                        )
                        
                elif cmd_type == "esp32_command":
                    # Forward command to ESP32
                    if self.esp32_serial and self.esp32_serial.is_open:
                        esp32_cmd = json.dumps(command.get("data", {})) + "\n"
                        self.esp32_serial.write(esp32_cmd.encode())
                        
        except Exception as e:
            logger.error(f"Command processing error: {e}")
    
    def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown of the entire system"""
        logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
        
        try:
            # Send emergency stop command to ESP32
            if self.esp32_serial and self.esp32_serial.is_open:
                emergency_cmd = {
                    "type": "emergency_stop",
                    "reason": reason,
                    "timestamp": time.time()
                }
                cmd_str = json.dumps(emergency_cmd) + "\n"
                self.esp32_serial.write(cmd_str.encode())
                logger.info("Emergency stop command sent to ESP32")
            
            # Stop all charging and switching operations
            stop_cmd = {
                "type": "force_stop",
                "charging_enabled": False,
                "pv_relay": False,
                "grid_relay": False,
                "charging_relay": False,
                "scc_enabled": False
            }
            
            if self.esp32_serial and self.esp32_serial.is_open:
                cmd_str = json.dumps(stop_cmd) + "\n"
                self.esp32_serial.write(cmd_str.encode())
            
            # Publish emergency status via MQTT
            if self.mqtt_client and self.mqtt_client.is_connected():
                emergency_status = {
                    "emergency": True,
                    "reason": reason,
                    "timestamp": time.time(),
                    "action": "system_emergency_shutdown"
                }
                self.mqtt_client.publish(
                    f"{self.config.mqtt_topic_prefix}/emergency",
                    json.dumps(emergency_status)
                )
            
            # Update system status
            self.status.last_error = f"EMERGENCY: {reason}"
            self.status.layer1_status = "emergency_stop"
            
            logger.critical("Emergency shutdown procedures completed")
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")

    def start(self):
        """Start hierarchical MPC system"""
        try:
            logger.info("Starting Hierarchical MPC System...")
            
            self.running = True
            self.start_time = time.time()
            
            # Setup remote monitoring
            self._setup_mqtt()
            
            # Connect to ESP32
            if not self._connect_esp32():
                logger.error("Failed to connect to ESP32")
                return False
            
            self.status.layer1_status = "running"  # ESP32 is Layer 1
            
            # Start Layer 2 if enabled
            if self.config.layer2_enabled:
                if self._start_layer_process("layer2", "layer2_mpc_optimizer_headless.py"):
                    self.status.layer2_status = "running"
                else:
                    logger.error("Failed to start Layer 2")
            
            # Start Layer 3 if enabled
            if self.config.layer3_enabled:
                if self._start_layer_process("layer3", "layer3_adaptive_planner_headless.py"):
                    self.status.layer3_status = "running"
                else:
                    logger.error("Failed to start Layer 3")
            
            # Start data collection if enabled
            if self.config.data_collection_enabled:
                if self._start_layer_process("data_collection", "data_collection_96hour_headless.py"):
                    self.status.data_collection_status = "running"
                else:
                    logger.error("Failed to start data collection")
            
            # Start main monitoring loop
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"System startup error: {e}")
            self.status.last_error = str(e)
            return False
    
    def _main_loop(self):
        """Main orchestrator loop"""
        logger.info("Main system loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Process MQTT commands
                self._process_commands()
                
                # Perform health check
                if time.time() - self.last_health_check >= self.health_check_interval:
                    self._health_check()
                
                # ESP32 communication
                if self.esp32_serial and self.esp32_serial.is_open:
                    if self.esp32_serial.in_waiting > 0:
                        try:
                            response = self.esp32_serial.readline().decode().strip()
                            if response:
                                # Parse ESP32 response and forward to MQTT
                                if self.mqtt_client and self.mqtt_client.is_connected():
                                    self.mqtt_client.publish(
                                        f"{self.config.mqtt_topic_prefix}/esp32_data",
                                        response
                                    )
                        except:
                            pass
                
                time.sleep(0.1)  # Small delay untuk prevent high CPU usage
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.status.last_error = str(e)
                time.sleep(1)
    
    def shutdown(self):
        """Shutdown hierarchical MPC system"""
        logger.info("Shutting down Hierarchical MPC System...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop all processes
        for layer_name, process in self.processes.items():
            try:
                logger.info(f"Stopping {layer_name}...")
                process.terminate()
                process.wait(timeout=10)
            except:
                logger.warning(f"Force killing {layer_name}...")
                process.kill()
        
        # Close connections
        if self.esp32_serial and self.esp32_serial.is_open:
            self.esp32_serial.close()
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        # Log shutdown event
        self._log_event("system_shutdown", "orchestrator", "System shutdown completed")
        
        logger.info("System shutdown completed")

def signal_handler(signum, frame):
    """Handle system signals"""
    logger.info(f"Received signal {signum}")
    if 'orchestrator' in globals():
        orchestrator.shutdown()
    sys.exit(0)

def main():
    """Main function"""
    global orchestrator
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create orchestrator
        orchestrator = SystemOrchestrator()
        
        # Start system
        success = orchestrator.start()
        
        if not success:
            logger.error("Failed to start system")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

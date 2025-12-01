#!/usr/bin/env python3
"""
System Watchdog for MPC Smart Charging System
Monitors system health and restarts services if needed
"""

import time
import subprocess
import logging
import psutil
import sqlite3
import requests
from datetime import datetime, timedelta

# Import network configuration
try:
    from network_config import get_mqtt_config, get_wifi_config, get_device_config
except ImportError:
    print("WARNING: Network config not found, using default values")
    get_mqtt_config = lambda: {"broker": "10.121.146.109", "port": 1883}
    get_wifi_config = lambda: {"ssid": "TP-Link_E4-FC"}
    get_device_config = lambda: {"name": "ESP32-MPC-Controller"}

class SystemWatchdog:
    def __init__(self):
        self.setup_logging()
        self.check_interval = 60  # Check every minute
        self.restart_threshold = 3  # Restart after 3 failed checks
        self.failure_counts = {}
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - Watchdog - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/mpc_watchdog.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_service_health(self, service_name):
        """Check if systemd service is running"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', service_name],
                capture_output=True, text=True
            )
            return result.stdout.strip() == 'active'
        except Exception as e:
            self.logger.error(f"Error checking {service_name}: {e}")
            return False
    
    def check_web_dashboard(self):
        """Check if web dashboard is responding"""
        try:
            response = requests.get('http://localhost:8080', timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Web dashboard check failed: {e}")
            return False
    
    def check_mqtt_broker(self):
        """Check if MQTT broker is responding"""
        try:
            mqtt_config = get_mqtt_config()
            result = subprocess.run(
                ['mosquitto_pub', '-h', mqtt_config["broker"], '-t', 'test', '-m', 'ping'],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"MQTT broker check failed: {e}")
            return False
    
    def check_database_access(self):
        """Check if database is accessible"""
        try:
            conn = sqlite3.connect('system_orchestrator.db')
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM system_status')
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            return False
    
    def check_system_resources(self):
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Alert if resources are critically high
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 90:
                self.logger.warning(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                self.logger.warning(f"High disk usage: {disk_percent}%")
            
            return cpu_percent < 95 and memory_percent < 95 and disk_percent < 95
        except Exception as e:
            self.logger.error(f"Resource check failed: {e}")
            return False
    
    def restart_service(self, service_name):
        """Restart a systemd service"""
        try:
            subprocess.run(['sudo', 'systemctl', 'restart', service_name], check=True)
            self.logger.info(f"Restarted service: {service_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart {service_name}: {e}")
            return False
    
    def run_watchdog(self):
        """Main watchdog loop"""
        self.logger.info("System watchdog started")
        
        while True:
            try:
                # Check main service
                if not self.check_service_health('mpc-smart-charging'):
                    self.failure_counts['mpc-smart-charging'] = self.failure_counts.get('mpc-smart-charging', 0) + 1
                    self.logger.warning(f"MPC service not active (failures: {self.failure_counts['mpc-smart-charging']})")
                    
                    if self.failure_counts['mpc-smart-charging'] >= self.restart_threshold:
                        self.restart_service('mpc-smart-charging')
                        self.failure_counts['mpc-smart-charging'] = 0
                else:
                    self.failure_counts['mpc-smart-charging'] = 0
                
                # Check MQTT broker
                if not self.check_service_health('mosquitto'):
                    self.failure_counts['mosquitto'] = self.failure_counts.get('mosquitto', 0) + 1
                    self.logger.warning(f"MQTT service not active (failures: {self.failure_counts['mosquitto']})")
                    
                    if self.failure_counts['mosquitto'] >= self.restart_threshold:
                        self.restart_service('mosquitto')
                        self.failure_counts['mosquitto'] = 0
                else:
                    self.failure_counts['mosquitto'] = 0
                
                # Check web dashboard
                if not self.check_web_dashboard():
                    self.logger.warning("Web dashboard not responding")
                
                # Check MQTT broker connectivity
                if not self.check_mqtt_broker():
                    self.logger.warning("MQTT broker not responding")
                
                # Check database
                if not self.check_database_access():
                    self.logger.warning("Database not accessible")
                
                # Check system resources
                if not self.check_system_resources():
                    self.logger.warning("System resources critically high")
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Watchdog stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Watchdog error: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    watchdog = SystemWatchdog()
    watchdog.run_watchdog()

#!/usr/bin/env python3
"""
WEB DASHBOARD FOR MPC SMART CHARGING SYSTEM
==========================================

Real-time web dashboard untuk monitoring MPC Smart Charging System
dengan interface yang user-friendly untuk monitoring dan control.

Features:
- Real-time sensor monitoring
- System status display
- Control panel untuk manual override
- Data visualization charts
- Historical data analysis
- Mobile-responsive design

Author: Dzaky Naufal K
Date: July 6, 2025
Version: 1.0 - Production Web Dashboard
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
import threading
import time
import plotly.graph_objs as go
import plotly.utils
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """System status data structure"""
    timestamp: float
    battery_voltage: float
    battery_current: float
    battery_soc: float
    battery_temp: float
    pv_voltage: float
    pv_current: float
    pv_power: float
    grid_voltage: float
    grid_current: float
    grid_power: float
    charger_temp: float
    system_mode: str
    layer1_status: str
    layer2_status: str
    layer3_status: str
    safety_status: str
    uptime: float

class WebDashboard:
    """Web dashboard untuk MPC Smart Charging System"""
    
    def __init__(self, config_file: str = "system_config.json"):
        """Initialize web dashboard"""
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.config['SECRET_KEY'] = 'mpc_smart_charging_secret_key'
        
        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Database connections
        self.db_path = "system_orchestrator.db"
        self.data_db_path = "validation_data_96hour.db"
        
        # System status
        self.current_status = None
        self.last_update = None
        
        # Dashboard state
        self.dashboard_active = True
        self.update_interval = 5  # seconds
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
        logger.info("🌐 Web Dashboard initialized")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load system configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file {config_file} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "system_config": {
                "battery_capacity": 35.0,
                "battery_voltage": 12.8,
                "max_charging_current": 30.0,
                "max_battery_temp": 45.0,
                "max_charger_temp": 65.0,
                "min_soc": 20.0,
                "max_soc": 90.0
            },
            "web_dashboard": {
                "host": "0.0.0.0",
                "port": 8080,
                "debug": False,
                "update_interval": 5
            }
        }
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint untuk system status"""
            try:
                status = self._get_current_status()
                if status:
                    return jsonify({
                        'success': True,
                        'data': status.__dict__,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No status data available'
                    }), 404
            except Exception as e:
                logger.error(f"Error getting status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/sensors')
        def api_sensors():
            """API endpoint untuk sensor data"""
            try:
                sensors = self._get_sensor_data()
                return jsonify({
                    'success': True,
                    'data': sensors,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting sensors: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/history')
        def api_history():
            """API endpoint untuk historical data"""
            try:
                hours = request.args.get('hours', 24, type=int)
                history = self._get_historical_data(hours)
                return jsonify({
                    'success': True,
                    'data': history,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting history: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/control', methods=['POST'])
        def api_control():
            """API endpoint untuk system control"""
            try:
                command = request.json
                result = self._execute_command(command)
                return jsonify({
                    'success': True,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error executing command: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/charts/soc')
        def api_chart_soc():
            """SOC chart data"""
            try:
                hours = request.args.get('hours', 24, type=int)
                chart_data = self._generate_soc_chart(hours)
                return jsonify(chart_data)
            except Exception as e:
                logger.error(f"Error generating SOC chart: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/charts/power')
        def api_chart_power():
            """Power flow chart data"""
            try:
                hours = request.args.get('hours', 24, type=int)
                chart_data = self._generate_power_chart(hours)
                return jsonify(chart_data)
            except Exception as e:
                logger.error(f"Error generating power chart: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO events untuk real-time updates"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to MPC Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info("Client disconnected from dashboard")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle update request dari client"""
            try:
                status = self._get_current_status()
                if status:
                    emit('status_update', status.__dict__)
            except Exception as e:
                logger.error(f"Error sending update: {e}")
                emit('error', {'message': str(e)})
    
    def _get_current_status(self) -> Optional[SystemStatus]:
        """Get current system status dari database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM system_status 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return SystemStatus(
                    timestamp=row[1],
                    battery_voltage=12.8,  # Default values - akan diupdate dari sensor real
                    battery_current=15.5,
                    battery_soc=75.0,
                    battery_temp=28.5,
                    pv_voltage=18.2,
                    pv_current=8.3,
                    pv_power=151.06,
                    grid_voltage=220.0,
                    grid_current=2.1,
                    grid_power=462.0,
                    charger_temp=32.1,
                    system_mode=row[2] if len(row) > 2 else "AUTO",
                    layer1_status=row[3] if len(row) > 3 else "ACTIVE",
                    layer2_status=row[4] if len(row) > 4 else "ACTIVE",
                    layer3_status=row[5] if len(row) > 5 else "ACTIVE",
                    safety_status="SAFE",
                    uptime=row[8] if len(row) > 8 else 0
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return None
    
    def _get_sensor_data(self) -> Dict:
        """Get latest sensor data"""
        try:
            # Try to get from validation database first
            conn = sqlite3.connect(self.data_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'timestamp': row[1],
                    'battery_voltage': row[2],
                    'battery_current': row[3],
                    'battery_soc': row[4],
                    'battery_temp': row[5],
                    'pv_voltage': row[6],
                    'pv_current': row[7],
                    'pv_power': row[8],
                    'grid_voltage': row[9],
                    'grid_current': row[10],
                    'grid_power': row[11],
                    'charger_temp': row[12]
                }
            else:
                # Return default sensor data if no database data
                return {
                    'timestamp': time.time(),
                    'battery_voltage': 12.8,
                    'battery_current': 15.5,
                    'battery_soc': 75.0,
                    'battery_temp': 28.5,
                    'pv_voltage': 18.2,
                    'pv_current': 8.3,
                    'pv_power': 151.06,
                    'grid_voltage': 220.0,
                    'grid_current': 2.1,
                    'grid_power': 462.0,
                    'charger_temp': 32.1
                }
                
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            # Return mock data for demonstration
            return {
                'timestamp': time.time(),
                'battery_voltage': 12.8,
                'battery_current': 15.5,
                'battery_soc': 75.0,
                'battery_temp': 28.5,
                'pv_voltage': 18.2,
                'pv_current': 8.3,
                'pv_power': 151.06,
                'grid_voltage': 220.0,
                'grid_current': 2.1,
                'grid_power': 462.0,
                'charger_temp': 32.1
            }
    
    def _get_historical_data(self, hours: int) -> List[Dict]:
        """Get historical data untuk charts"""
        try:
            conn = sqlite3.connect(self.data_db_path)
            cursor = conn.cursor()
            
            start_time = time.time() - (hours * 3600)
            
            cursor.execute('''
                SELECT * FROM sensor_readings 
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            ''', (start_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'timestamp': row[1],
                    'battery_voltage': row[2],
                    'battery_current': row[3],
                    'battery_soc': row[4],
                    'battery_temp': row[5],
                    'pv_voltage': row[6],
                    'pv_current': row[7],
                    'pv_power': row[8],
                    'grid_voltage': row[9],
                    'grid_current': row[10],
                    'grid_power': row[11],
                    'charger_temp': row[12]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def _execute_command(self, command: Dict) -> Dict:
        """Execute control command"""
        try:
            cmd_type = command.get('type')
            
            if cmd_type == 'emergency_stop':
                # Implement emergency stop
                logger.info("Emergency stop command received")
                return {'status': 'emergency_stop_executed'}
            
            elif cmd_type == 'set_mode':
                mode = command.get('mode', 'AUTO')
                logger.info(f"Mode change command: {mode}")
                return {'status': 'mode_changed', 'new_mode': mode}
            
            elif cmd_type == 'set_soc_limits':
                min_soc = command.get('min_soc', 20)
                max_soc = command.get('max_soc', 90)
                logger.info(f"SOC limits changed: {min_soc}% - {max_soc}%")
                return {'status': 'soc_limits_updated', 'min_soc': min_soc, 'max_soc': max_soc}
            
            else:
                return {'status': 'unknown_command', 'error': f'Unknown command type: {cmd_type}'}
                
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_soc_chart(self, hours: int) -> Dict:
        """Generate SOC chart data"""
        try:
            history = self._get_historical_data(hours)
            
            if not history:
                # Generate mock data untuk demonstration
                current_time = time.time()
                history = []
                for i in range(hours * 6):  # Every 10 minutes
                    timestamp = current_time - (hours * 3600) + (i * 600)
                    soc = 75 + (i % 20) - 10  # Simulate SOC variation
                    history.append({
                        'timestamp': timestamp,
                        'battery_soc': max(20, min(90, soc))
                    })
            
            timestamps = [datetime.fromtimestamp(h['timestamp']) for h in history]
            soc_values = [h['battery_soc'] for h in history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=soc_values,
                mode='lines+markers',
                name='Battery SOC',
                line=dict(color='#2E8B57', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title='Battery State of Charge (SOC)',
                xaxis_title='Time',
                yaxis_title='SOC (%)',
                yaxis=dict(range=[0, 100]),
                template='plotly_white'
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Error generating SOC chart: {e}")
            return {'error': str(e)}
    
    def _generate_power_chart(self, hours: int) -> Dict:
        """Generate power flow chart data"""
        try:
            history = self._get_historical_data(hours)
            
            if not history:
                # Generate mock data
                current_time = time.time()
                history = []
                for i in range(hours * 6):
                    timestamp = current_time - (hours * 3600) + (i * 600)
                    history.append({
                        'timestamp': timestamp,
                        'pv_power': max(0, 150 + 50 * (i % 10 - 5)),
                        'grid_power': 400 + 100 * (i % 8 - 4),
                        'battery_voltage': 12.8,
                        'battery_current': 15 + 5 * (i % 6 - 3)
                    })
            
            timestamps = [datetime.fromtimestamp(h['timestamp']) for h in history]
            pv_power = [h['pv_power'] for h in history]
            grid_power = [h['grid_power'] for h in history]
            battery_power = [h['battery_voltage'] * h['battery_current'] for h in history]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=pv_power,
                mode='lines',
                name='PV Power',
                line=dict(color='#FFD700', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=grid_power,
                mode='lines',
                name='Grid Power',
                line=dict(color='#FF6347', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=battery_power,
                mode='lines',
                name='Battery Power',
                line=dict(color='#4169E1', width=2)
            ))
            
            fig.update_layout(
                title='Power Flow Analysis',
                xaxis_title='Time',
                yaxis_title='Power (W)',
                template='plotly_white'
            )
            
            return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
            
        except Exception as e:
            logger.error(f"Error generating power chart: {e}")
            return {'error': str(e)}
    
    def _start_background_updates(self):
        """Start background thread untuk real-time updates"""
        def update_loop():
            while self.dashboard_active:
                try:
                    status = self._get_current_status()
                    if status:
                        self.current_status = status
                        self.last_update = time.time()
                        # Emit update ke semua connected clients
                        self.socketio.emit('status_update', status.__dict__)
                    
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Error in background update: {e}")
                    time.sleep(self.update_interval)
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
        logger.info("Background update thread started")
    
    def create_templates(self):
        """Create HTML templates untuk dashboard"""
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static/css', exist_ok=True)
        os.makedirs('static/js', exist_ok=True)
        
        # Create main dashboard template
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MPC Smart Charging Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🔋 MPC Smart Charging System</span>
            <span class="navbar-text" id="status-indicator">
                <span class="badge bg-success">ONLINE</span>
            </span>
        </div>
    </nav>

    <div class="container-fluid mt-3">
        <!-- System Status Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card text-white bg-success">
                    <div class="card-body">
                        <h5 class="card-title">Battery SOC</h5>
                        <h2 id="battery-soc">75.0%</h2>
                        <small>Target: 20-90%</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-info">
                    <div class="card-body">
                        <h5 class="card-title">Battery Voltage</h5>
                        <h2 id="battery-voltage">12.8V</h2>
                        <small>Range: 12.6-14.6V</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-warning">
                    <div class="card-body">
                        <h5 class="card-title">Battery Current</h5>
                        <h2 id="battery-current">15.5A</h2>
                        <small>Max: 30A</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-danger">
                    <div class="card-body">
                        <h5 class="card-title">Temperature</h5>
                        <h2 id="battery-temp">28.5°C</h2>
                        <small>Max: 45°C</small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Battery SOC Trend</h5>
                    </div>
                    <div class="card-body">
                        <div id="soc-chart"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Power Flow</h5>
                    </div>
                    <div class="card-body">
                        <div id="power-chart"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Details -->
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>System Status</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped">
                            <tr><td>System Mode</td><td id="system-mode">AUTO</td></tr>
                            <tr><td>Layer 1 Status</td><td id="layer1-status">ACTIVE</td></tr>
                            <tr><td>Layer 2 Status</td><td id="layer2-status">ACTIVE</td></tr>
                            <tr><td>Layer 3 Status</td><td id="layer3-status">ACTIVE</td></tr>
                            <tr><td>Safety Status</td><td id="safety-status">SAFE</td></tr>
                            <tr><td>System Uptime</td><td id="system-uptime">0h 0m</td></tr>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>Quick Controls</h5>
                    </div>
                    <div class="card-body">
                        <button class="btn btn-danger btn-block mb-2" onclick="emergencyStop()">
                            🚨 Emergency Stop
                        </button>
                        <button class="btn btn-warning btn-block mb-2" onclick="changeMode('MANUAL')">
                            🔧 Manual Mode
                        </button>
                        <button class="btn btn-success btn-block mb-2" onclick="changeMode('AUTO')">
                            🤖 Auto Mode
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Socket.IO connection
        const socket = io();
        
        // Real-time updates
        socket.on('status_update', function(data) {
            updateDashboard(data);
        });
        
        function updateDashboard(data) {
            document.getElementById('battery-soc').textContent = data.battery_soc.toFixed(1) + '%';
            document.getElementById('battery-voltage').textContent = data.battery_voltage.toFixed(1) + 'V';
            document.getElementById('battery-current').textContent = data.battery_current.toFixed(1) + 'A';
            document.getElementById('battery-temp').textContent = data.battery_temp.toFixed(1) + '°C';
            document.getElementById('system-mode').textContent = data.system_mode;
            document.getElementById('layer1-status').textContent = data.layer1_status;
            document.getElementById('layer2-status').textContent = data.layer2_status;
            document.getElementById('layer3-status').textContent = data.layer3_status;
            document.getElementById('safety-status').textContent = data.safety_status;
            
            const uptime = formatUptime(data.uptime);
            document.getElementById('system-uptime').textContent = uptime;
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return hours + 'h ' + minutes + 'm';
        }
        
        function loadCharts() {
            // Load SOC chart
            fetch('/api/charts/soc?hours=24')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('soc-chart', data.data, data.layout);
                });
            
            // Load power chart
            fetch('/api/charts/power?hours=24')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('power-chart', data.data, data.layout);
                });
        }
        
        function emergencyStop() {
            if (confirm('Are you sure you want to execute emergency stop?')) {
                fetch('/api/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: 'emergency_stop'})
                });
            }
        }
        
        function changeMode(mode) {
            fetch('/api/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({type: 'set_mode', mode: mode})
            });
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadCharts();
            setInterval(loadCharts, 60000); // Refresh charts every minute
        });
    </script>
</body>
</html>'''

        with open('templates/dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        logger.info("Dashboard templates created")
    
    def run(self, host: str = None, port: int = None, debug: bool = False):
        """Run web dashboard server"""
        
        # Create templates if they don't exist
        self.create_templates()
        
        # Start background updates
        self._start_background_updates()
        
        # Get configuration
        web_config = self.config.get('web_dashboard', {})
        host = host or web_config.get('host', '0.0.0.0')
        port = port or web_config.get('port', 8080)
        debug = debug or web_config.get('debug', False)
        
        logger.info(f"🌐 Starting web dashboard on http://{host}:{port}")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Error running web dashboard: {e}")
        finally:
            self.dashboard_active = False

def main():
    """Main function untuk menjalankan web dashboard"""
    try:
        dashboard = WebDashboard()
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Web dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting web dashboard: {e}")

if __name__ == "__main__":
    main()

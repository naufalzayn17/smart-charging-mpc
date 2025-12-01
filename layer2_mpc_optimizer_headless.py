#!/usr/bin/env python3
"""
LAYER 2 MPC OPTIMIZER - HEADLESS HIERARCHICAL SYSTEM
====================================================

Layer 2: Slow MPC Controller dengan 6-hour horizon
- Headless operation untuk sistem standalone
- LiFePO4 4S battery model dengan proper dynamics
- Grid tariff optimization (Indonesian TOU simulation)
- Remote monitoring via MQTT/WiFi
- 96-hour data collection capability

Author: Dzaky Naufal K
Date: July 2, 2025
Version: 3.0 - Headless Hierarchical MPC
"""

import sys
import os
import time
import json
import logging
import numpy as np
import cvxpy as cp
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import paho.mqtt.client as mqtt
import serial
from dataclasses import dataclass
import math

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging for headless operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Layer2-MPC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mpc_layer2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BatteryState:
    """LiFePO4 4S Battery State"""
    soc: float          # State of charge (0-1)
    voltage: float      # Terminal voltage (V)
    current: float      # Current (A, positive = charging)
    temperature: float  # Temperature (°C)
    capacity_ah: float  # Capacity (Ah)
    internal_resistance: float  # Internal resistance (Ohm)

@dataclass
class SystemState:
    """Complete system state"""
    battery: BatteryState
    pv_power: float     # PV power (W)
    grid_power: float   # Grid power (W)
    load_power: float   # Load power (W)
    grid_available: bool # Grid availability
    grid_price: float   # Current grid price (Rp/kWh)
    timestamp: datetime

@dataclass
class MPCControl:
    """MPC Control outputs"""
    use_pv: bool
    use_grid: bool
    charge_current_ref: float
    predicted_soc: List[float]
    predicted_cost: float
    solver_time: float
    feasible: bool

class LiFePO4BatteryModel:
    """LiFePO4 4S Battery Model untuk MPC"""
    
    def __init__(self):
        self.capacity_ah = 35.0          # 35Ah+ capacity
        self.voltage_nominal = 12.8      # 12.8V nominal voltage
        self.voltage_min = 12.6          # Minimum operational voltage
        self.voltage_max = 13.5          # Maximum operational voltage
        self.voltage_charge = 14.6       # Charging voltage
        self.current_max_charge = 30.0   # MPPT 30A
        self.current_max_discharge = 30.0 # BMS limit (conservative)
        self.efficiency_charge = 0.95    # Charging efficiency
        self.efficiency_discharge = 0.98 # Discharge efficiency
        
        logger.info(f"LiFePO4 Battery Model: {self.capacity_ah}Ah, {self.voltage_min}-{self.voltage_max}V")
    
    def get_ocv(self, soc: float) -> float:
        """Open Circuit Voltage vs SOC curve for LiFePO4 4P6S 12V"""
        soc = np.clip(soc, 0.0, 1.0)
        
        # LiFePO4 4P6S 12V system voltage curve
        if soc >= 0.95:
            return 13.5  # Maximum operational
        elif soc >= 0.85:
            return 13.4 + (soc - 0.85) * 1.0  # 85-95%
        elif soc >= 0.70:
            return 13.2 + (soc - 0.70) * 1.33  # 70-85%
        elif soc >= 0.50:
            return 13.0 + (soc - 0.50) * 1.0   # 50-70%
        elif soc >= 0.30:
            return 12.8 + (soc - 0.30) * 1.0   # 30-50%
        elif soc >= 0.20:
            return 12.6 + (soc - 0.20) * 2.0   # 20-30%
        elif soc >= 0.10:
            return 12.0 + (soc - 0.10) * 6.0   # 10-20%
        elif soc >= 0.05:
            return 11.5 + (soc - 0.05) * 10.0  # 5-10%
        else:
            return 10.0 + soc * 30.0  # 0-5% (steep drop)
    
    def get_internal_resistance(self, soc: float, temperature: float = 25.0) -> float:
        """Internal resistance vs SOC and temperature"""
        soc = np.clip(soc, 0.0, 1.0)
        
        # Base resistance (higher at extreme SOC)
        if soc < 0.1:
            r_base = 0.08  # High resistance at low SOC
        elif soc > 0.9:
            r_base = 0.06  # Slightly higher at high SOC
        else:
            r_base = 0.05  # Normal resistance
        
        # Temperature compensation (resistance increases at low temp)
        temp_factor = 1.0 + max(0, (25 - temperature) * 0.01)
        
        return r_base * temp_factor
    
    def update_state(self, state: BatteryState, current: float, dt: float) -> BatteryState:
        """Update battery state with current integration"""
        # Apply efficiency
        if current > 0:  # Charging
            effective_current = current * self.efficiency_charge
        else:  # Discharging
            effective_current = current / self.efficiency_discharge
        
        # Update SOC with coulomb counting
        soc_new = state.soc + (effective_current * dt) / (self.capacity_ah * 3600)
        soc_new = np.clip(soc_new, 0.0, 1.0)
        
        # Update voltage with I*R drop
        ocv = self.get_ocv(soc_new)
        resistance = self.get_internal_resistance(soc_new, state.temperature)
        voltage_new = ocv - current * resistance
        
        # Create new state
        new_state = BatteryState(
            soc=soc_new,
            voltage=voltage_new,
            current=current,
            temperature=state.temperature,  # Temperature model could be added
            capacity_ah=self.capacity_ah,
            internal_resistance=resistance
        )
        
        return new_state

class GridTariffModel:
    """Indonesian electricity tariff simulation"""
    
    def __init__(self):
        # Simplified Indonesian TOU tariff (PLN)
        self.tariff_peak = 1467     # Rp/kWh peak hours (17:00-22:00)
        self.tariff_offpeak = 1035  # Rp/kWh off-peak hours
        self.tariff_basic = 1200    # Rp/kWh basic rate
        
        logger.info(f"Grid Tariff: Peak={self.tariff_peak}, Off-peak={self.tariff_offpeak} Rp/kWh")
    
    def get_electricity_price(self, hour: int) -> float:
        """Get electricity price based on time of day"""
        if 17 <= hour <= 22:  # Peak hours (5 PM - 10 PM)
            return self.tariff_peak
        else:  # Off-peak hours
            return self.tariff_offpeak
    
    def get_price_prediction(self, start_hour: int, horizon_hours: int) -> List[float]:
        """Get price prediction for MPC horizon"""
        prices = []
        for h in range(horizon_hours):
            hour = (start_hour + h) % 24
            prices.append(self.get_electricity_price(hour))
        return prices

class Layer2MPCOptimizer:
    """Layer 2 MPC Optimizer - 6 hour horizon"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.Layer2MPC")
        
        # MPC Parameters
        self.horizon_hours = 6          # 6-hour prediction horizon
        self.control_horizon_hours = 2  # 2-hour control horizon
        self.dt_minutes = 10           # 10-minute time steps
        self.n_steps = int(self.horizon_hours * 60 / self.dt_minutes)  # 36 steps
        self.n_control = int(self.control_horizon_hours * 60 / self.dt_minutes)  # 12 steps
        
        # Models
        self.battery_model = LiFePO4BatteryModel()
        self.tariff_model = GridTariffModel()
        
        # MPC weights (can be adapted)
        self.weights = {
            'efficiency': 1.0,      # Prefer PV over grid
            'cost': 0.8,           # Minimize electricity cost
            'battery_health': 0.6, # Avoid extreme SOC and currents
            'comfort': 0.4         # Maintain SOC within comfort zone
        }
        
        # Database for logging
        self.db_path = '/var/log/mpc_layer2_data.db'
        self.init_database()
        
        self.logger.info(f"Layer 2 MPC initialized: {self.horizon_hours}h horizon, {self.dt_minutes}min steps")
    
    def init_database(self):
        """Initialize database for Layer 2 logging"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # MPC performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mpc_layer2_performance (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    solver_time_ms REAL,
                    objective_value REAL,
                    prediction_horizon_hours INTEGER,
                    feasible BOOLEAN,
                    weights_efficiency REAL,
                    weights_cost REAL,
                    weights_battery_health REAL,
                    weights_comfort REAL
                )
            ''')
            
            # Battery predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS battery_predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    step_ahead INTEGER,
                    predicted_soc REAL,
                    predicted_voltage REAL,
                    predicted_current REAL,
                    predicted_power REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Layer 2 database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def setup_mpc_problem(self, initial_state: SystemState, 
                         pv_forecast: List[float], 
                         load_forecast: List[float]) -> Tuple[cp.Problem, Dict]:
        """Setup MPC optimization problem"""
        
        # Decision variables
        soc = cp.Variable(self.n_steps + 1)              # SOC trajectory
        current = cp.Variable(self.n_steps)              # Battery current
        use_pv = cp.Variable(self.n_steps, boolean=True) # PV usage (binary)
        use_grid = cp.Variable(self.n_steps, boolean=True) # Grid usage (binary)
        voltage = cp.Variable(self.n_steps)              # Battery voltage
        
        # Grid prices for horizon
        current_hour = initial_state.timestamp.hour
        grid_prices = self.tariff_model.get_price_prediction(current_hour, self.n_steps)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(soc[0] == initial_state.battery.soc)
        
        # SOC dynamics and constraints
        dt_hours = self.dt_minutes / 60.0
        for k in range(self.n_steps):
            # SOC update (simplified)
            constraints.append(
                soc[k+1] == soc[k] + (current[k] * dt_hours) / self.battery_model.capacity_ah
            )
            
            # SOC bounds
            constraints.append(soc[k+1] >= 0.15)  # Minimum 15% SOC
            constraints.append(soc[k+1] <= 0.95)  # Maximum 95% SOC
            
            # Current bounds
            constraints.append(current[k] >= -self.battery_model.current_max_discharge)
            constraints.append(current[k] <= self.battery_model.current_max_charge)
            
            # Voltage constraint (simplified)
            constraints.append(voltage[k] >= self.battery_model.voltage_min)
            constraints.append(voltage[k] <= self.battery_model.voltage_max)
            
            # Power source constraints (mutual exclusion with relaxation)
            constraints.append(use_pv[k] + use_grid[k] <= 1)
            
            # PV availability constraint
            if k < len(pv_forecast) and pv_forecast[k] < 50:  # Insufficient PV
                constraints.append(use_pv[k] == 0)
        
        # Objective function
        cost_terms = []
        efficiency_terms = []
        battery_health_terms = []
        comfort_terms = []
        
        for k in range(self.n_steps):
            # Cost term (grid usage cost)
            grid_power_k = cp.maximum(0, current[k] * voltage[k])  # Only charging from grid
            cost_terms.append(use_grid[k] * grid_power_k * grid_prices[k] * dt_hours / 1000)
            
            # Efficiency term (prefer PV)
            efficiency_terms.append(use_pv[k] * 100 - use_grid[k] * 50)
            
            # Battery health term (avoid extreme currents and SOC)
            battery_health_terms.append(
                -cp.square(current[k] / self.battery_model.current_max_charge) * 10 -
                cp.maximum(0, soc[k+1] - 0.9) * 100 -  # Penalty for high SOC
                cp.maximum(0, 0.2 - soc[k+1]) * 100    # Penalty for low SOC
            )
            
            # Comfort term (maintain SOC in comfortable range)
            comfort_terms.append(-cp.square(soc[k+1] - 0.6) * 50)  # Target 60% SOC
        
        # Combined objective
        objective = (
            self.weights['efficiency'] * cp.sum(efficiency_terms) -
            self.weights['cost'] * cp.sum(cost_terms) +
            self.weights['battery_health'] * cp.sum(battery_health_terms) +
            self.weights['comfort'] * cp.sum(comfort_terms)
        )
        
        # Create problem
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        variables = {
            'soc': soc,
            'current': current,
            'use_pv': use_pv,
            'use_grid': use_grid,
            'voltage': voltage
        }
        
        return problem, variables
    
    def solve_mpc(self, system_state: SystemState, 
                  pv_forecast: Optional[List[float]] = None,
                  load_forecast: Optional[List[float]] = None) -> MPCControl:
        """Solve Layer 2 MPC optimization"""
        
        start_time = time.time()
        
        # Default forecasts if not provided
        if pv_forecast is None:
            pv_forecast = [system_state.pv_power] * self.n_steps
        if load_forecast is None:
            load_forecast = [system_state.load_power] * self.n_steps
        
        try:
            # Setup and solve MPC problem
            problem, variables = self.setup_mpc_problem(system_state, pv_forecast, load_forecast)
            
            # Solve with timeout
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=1000)
            
            solver_time = (time.time() - start_time) * 1000  # ms
            
            if problem.status in ["infeasible", "unbounded"]:
                self.logger.warning(f"MPC problem {problem.status}")
                return self._get_fallback_control(system_state, solver_time)
            
            # Extract solution
            soc_trajectory = variables['soc'].value
            current_trajectory = variables['current'].value
            use_pv_solution = variables['use_pv'].value
            use_grid_solution = variables['use_grid'].value
            
            # First control action (receding horizon)
            control = MPCControl(
                use_pv=bool(use_pv_solution[0] > 0.5),
                use_grid=bool(use_grid_solution[0] > 0.5),
                charge_current_ref=float(current_trajectory[0]),
                predicted_soc=soc_trajectory.tolist() if soc_trajectory is not None else [],
                predicted_cost=float(problem.value) if problem.value is not None else 0.0,
                solver_time=solver_time,
                feasible=True
            )
            
            self.logger.info(f"MPC solved in {solver_time:.1f}ms: PV={control.use_pv}, Grid={control.use_grid}, I={control.charge_current_ref:.2f}A")
            
            # Log performance
            self._log_mpc_performance(system_state, control, problem.value)
            
            return control
            
        except Exception as e:
            solver_time = (time.time() - start_time) * 1000
            self.logger.error(f"MPC solver error: {e}")
            return self._get_fallback_control(system_state, solver_time)
    
    def _get_fallback_control(self, system_state: SystemState, solver_time: float) -> MPCControl:
        """Fallback control when MPC fails"""
        # Simple rule-based fallback
        use_pv = system_state.pv_power > 100 and system_state.battery.soc < 0.9
        use_grid = system_state.battery.soc < 0.3 and not use_pv
        
        if system_state.battery.soc < 0.2:
            charge_current = 10.0  # Conservative charging
        elif system_state.battery.soc > 0.9:
            charge_current = 0.0   # Stop charging
        else:
            charge_current = 5.0   # Maintenance charging
        
        return MPCControl(
            use_pv=use_pv,
            use_grid=use_grid,
            charge_current_ref=charge_current,
            predicted_soc=[],
            predicted_cost=0.0,
            solver_time=solver_time,
            feasible=False
        )
    
    def _log_mpc_performance(self, system_state: SystemState, control: MPCControl, objective_value: float):
        """Log MPC performance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mpc_layer2_performance 
                (timestamp, solver_time_ms, objective_value, prediction_horizon_hours, feasible,
                 weights_efficiency, weights_cost, weights_battery_health, weights_comfort)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                control.solver_time,
                objective_value if objective_value is not None else 0.0,
                self.horizon_hours,
                control.feasible,
                self.weights['efficiency'],
                self.weights['cost'],
                self.weights['battery_health'],
                self.weights['comfort']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Performance logging error: {e}")

# Performance target validation
PERFORMANCE_TARGETS = {
    'pv_utilization': 0.85,    # Use 85% of available PV
    'grid_reduction': 0.60,    # 60% less grid vs baseline  
    'soc_range': (20, 90),     # Keep SOC in healthy range (20-90%)
    'switching_frequency': 10, # Max 10 switches per day
    'system_efficiency': 0.88, # Overall system efficiency
    'cost_reduction': 0.40,    # 40% cost saving vs baseline
    'mpc_solver_time': 5.0     # Max 5 seconds solver time
}

def main():
    """Main function for headless Layer 2 operation"""
    logger.info("🚀 Starting Layer 2 MPC Optimizer - Headless Mode")
    
    try:
        # Initialize Layer 2 MPC
        mpc_optimizer = Layer2MPCOptimizer()
        
        # Main control loop
        while True:
            # This would be replaced with actual system state acquisition
            # For now, create a dummy state for testing
            dummy_state = SystemState(
                battery=BatteryState(
                    soc=0.6,
                    voltage=12.8,
                    current=5.0,
                    temperature=25.0,
                    capacity_ah=35.0,
                    internal_resistance=0.05
                ),
                pv_power=200.0,
                grid_power=0.0,
                load_power=800.0,
                grid_available=True,
                grid_price=1200.0,
                timestamp=datetime.now()
            )
            
            # Solve MPC
            control = mpc_optimizer.solve_mpc(dummy_state)
            
            # Log control action
            logger.info(f"Control: PV={control.use_pv}, Grid={control.use_grid}, Current={control.charge_current_ref:.2f}A")
            
            # Sleep for control interval (Layer 2 runs every hour)
            time.sleep(3600)  # 1 hour
            
    except KeyboardInterrupt:
        logger.info("Layer 2 MPC stopped by user")
    except Exception as e:
        logger.error(f"Layer 2 MPC error: {e}")
        raise

if __name__ == "__main__":
    main()

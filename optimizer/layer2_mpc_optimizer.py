#!/usr/bin/env python3
"""
LAYER 2 MPC OPTIMIZER
====================

Model Predictive Control optimizer untuk hierarchical charging system.
Implementasi MPC dengan objective function multi-criteria dan constraints.

Author: Dzaky Naufal K
Date: July 2, 2025
Version: 1.0 - MPC Core Implementation
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import optimization libraries
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    print("⚠️ CVXPY not available, using scipy.optimize fallback")
    CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    print("❌ scipy.optimize not available")
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdaptiveWeightTuner:
    """
    Adaptive weight tuning for MPC objective function
    Uses simple gradient-free optimization based on performance metrics
    """
    
    def __init__(self, initial_weights: Dict[str, float] = None):
        """
        Initialize adaptive weight tuner
        
        Args:
            initial_weights: Initial weight values
        """
        self.weights = initial_weights or {
            'economic': 1.0,
            'soc_deviation': 2.0,
            'switching': 0.5,
            'grid_usage': 0.8,
            'safety': 3.0
        }
        
        self.performance_history = []
        self.weight_history = []
        self.adaptation_rate = 0.1
        self.adaptation_interval = 50  # Adapt every 50 cycles
        self.cycle_count = 0
        
        # Performance targets
        self.target_efficiency = 0.85
        self.target_switching_freq = 4.0  # switches per hour
        self.target_soc_stability = 5.0   # % variation
        
        logger.info("Adaptive Weight Tuner initialized")
        logger.info(f"Initial weights: {self.weights}")
    
    def update_weights(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Update MPC weights based on performance metrics
        
        Args:
            performance_metrics: Dict with performance data
            
        Returns:
            Updated weight dictionary
        """
        try:
            self.cycle_count += 1
            self.performance_history.append(performance_metrics.copy())
            
            # Only adapt weights periodically
            if self.cycle_count % self.adaptation_interval != 0:
                return self.weights.copy()
            
            # Calculate recent performance averages
            recent_metrics = self._get_recent_performance()
            
            # Adaptive weight adjustments
            weight_updates = {}
            
            # Economic efficiency tuning
            if recent_metrics.get('avg_efficiency', 0.85) < self.target_efficiency:
                weight_updates['economic'] = self.adaptation_rate
                weight_updates['grid_usage'] = self.adaptation_rate * 0.5
                logger.info(f"Low efficiency ({recent_metrics['avg_efficiency']:.3f}), increasing economic weights")
            
            # Switching frequency tuning
            if recent_metrics.get('switching_frequency', 2.0) > self.target_switching_freq:
                weight_updates['switching'] = self.adaptation_rate
                logger.info(f"High switching frequency ({recent_metrics['switching_frequency']:.1f}), increasing switching penalty")
            
            # SOC stability tuning
            if recent_metrics.get('soc_variation', 3.0) > self.target_soc_stability:
                weight_updates['soc_deviation'] = self.adaptation_rate
                logger.info(f"High SOC variation ({recent_metrics['soc_variation']:.1f}%), increasing SOC weight")
            
            # Safety response tuning
            if recent_metrics.get('safety_violations', 0) > 0:
                weight_updates['safety'] = self.adaptation_rate * 2.0
                logger.warning(f"Safety violations detected, increasing safety weight")
            
            # Apply weight updates
            for key, update in weight_updates.items():
                if key in self.weights:
                    self.weights[key] *= (1.0 + update)
            
            # Normalize weights to maintain reasonable scale
            self._normalize_weights()
            
            # Log weight changes
            self.weight_history.append(self.weights.copy())
            logger.info(f"Updated MPC weights: {self.weights}")
            
            return self.weights.copy()
            
        except Exception as e:
            logger.error(f"Weight adaptation error: {e}")
            return self.weights.copy()
    
    def _get_recent_performance(self, window_size: int = 10) -> Dict[str, float]:
        """Get average performance metrics from recent history"""
        if len(self.performance_history) < window_size:
            recent_data = self.performance_history
        else:
            recent_data = self.performance_history[-window_size:]
        
        if not recent_data:
            return {}
        
        # Calculate averages
        avg_metrics = {}
        keys = recent_data[0].keys()
        
        for key in keys:
            values = [data.get(key, 0) for data in recent_data if key in data]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _normalize_weights(self):
        """Normalize weights to maintain reasonable scale"""
        # Prevent weights from growing too large
        max_weight = 10.0
        for key in self.weights:
            self.weights[key] = min(self.weights[key], max_weight)
        
        # Maintain relative scale (total sum around 5-8)
        total_weight = sum(self.weights.values())
        target_total = 6.0
        
        if total_weight > target_total * 1.5:  # If weights are too large
            scale_factor = target_total / total_weight
            for key in self.weights:
                self.weights[key] *= scale_factor
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weight values"""
        return self.weights.copy()
    
    def reset_weights(self, new_weights: Dict[str, float] = None):
        """Reset weights to initial or specified values"""
        if new_weights:
            self.weights = new_weights.copy()
        else:
            self.weights = {
                'economic': 1.0,
                'soc_deviation': 2.0,
                'switching': 0.5,
                'grid_usage': 0.8,
                'safety': 3.0
            }
        
        self.performance_history.clear()
        self.weight_history.clear()
        self.cycle_count = 0
        
        logger.info(f"Weights reset to: {self.weights}")
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and statistics"""
        return {
            'cycle_count': self.cycle_count,
            'current_weights': self.weights.copy(),
            'performance_history_length': len(self.performance_history),
            'weight_history_length': len(self.weight_history),
            'last_adaptation_cycle': (self.cycle_count // self.adaptation_interval) * self.adaptation_interval
        }

@dataclass
class MPCParameters:
    """MPC configuration parameters"""
    horizon: int = 6  # Prediction horizon (hours)
    control_horizon: int = 2  # Control horizon (hours)
    dt: float = 1.0  # Time step (hours)
    
    # Objective weights
    economic_weight: float = 1.0
    soc_deviation_weight: float = 2.0
    switching_penalty_weight: float = 0.5
    grid_usage_penalty_weight: float = 0.8
    safety_margin_weight: float = 3.0
    
    # System constraints
    soc_min: float = 20.0
    soc_max: float = 90.0
    current_max: float = 30.0
    temp_max: float = 45.0
    grid_power_max: float = 3000.0
    switching_max_per_hour: int = 2

@dataclass
class MPCResult:
    """MPC optimization result"""
    success: bool
    optimal_controls: np.ndarray
    predicted_states: np.ndarray
    objective_value: float
    solver_time: float
    solver_status: str
    economic_cost: float
    constraints_satisfied: bool

@dataclass
class SystemState:
    """Current system state for MPC"""
    soc: float
    battery_voltage: float
    battery_current: float
    battery_temp: float
    pv_power: float
    load_power: float
    grid_power: float
    electricity_price: float
    timestamp: float

@dataclass
class PredictionData:
    """Prediction data for MPC horizon"""
    pv_forecast: np.ndarray  # PV power forecast [W]
    load_forecast: np.ndarray  # Load power forecast [W]
    price_forecast: np.ndarray  # Electricity price forecast [$/kWh]
    temperature_forecast: np.ndarray  # Ambient temperature [°C]

class Layer2MPCOptimizer:
    """
    Layer 2 MPC Optimizer
    
    Objective Function:
    J = w1*economic_cost + w2*soc_deviation + w3*switching_penalty + 
        w4*grid_usage + w5*safety_margin
    
    State Vector: [SOC, Battery_Temp, Battery_Voltage]
    Control Vector: [Charging_Current, Switching_Mode]
    Disturbance: [PV_Power, Load_Power, Ambient_Temp]
    """
    
    def __init__(self, params: MPCParameters = None, enable_adaptive_weights: bool = True):
        """Initialize MPC optimizer"""
        self.params = params or MPCParameters()
        
        # Adaptive weight tuning
        self.enable_adaptive_weights = enable_adaptive_weights
        if self.enable_adaptive_weights:
            initial_weights = {
                'economic': self.params.economic_weight,
                'soc_deviation': self.params.soc_deviation_weight,
                'switching': self.params.switching_penalty_weight,
                'grid_usage': self.params.grid_usage_penalty_weight,
                'safety': self.params.safety_margin_weight
            }
            self.weight_tuner = AdaptiveWeightTuner(initial_weights)
            logger.info("MPC Optimizer initialized with adaptive weight tuning")
        else:
            self.weight_tuner = None
            logger.info("MPC Optimizer initialized with fixed weights")
        
        # System model parameters
        self.battery_capacity = 35.0  # Ah
        self.battery_voltage_nominal = 12.8  # V
        self.charging_efficiency = 0.95
        self.thermal_resistance = 0.1  # °C/W
        self.thermal_capacitance = 500.0  # J/°C
        
        # State space matrices (simplified linear model)
        self._setup_state_space_model()
        
        # Optimization variables
        self.solver_type = "cvxpy" if CVXPY_AVAILABLE else "scipy"
        
        # Performance tracking
        self.optimization_history = []
        self.total_optimizations = 0
        self.total_solver_time = 0.0
        
        logger.info(f"MPC Optimizer initialized with {self.solver_type} solver")
        logger.info(f"Horizon: {self.params.horizon}h, Control: {self.params.control_horizon}h")
    
    def _setup_state_space_model(self):
        """Setup linear state-space model for MPC"""
        # State: [SOC, Battery_Temp, Battery_Voltage]
        # Control: [Charging_Current, Switching_Mode] 
        # Disturbance: [PV_Power, Load_Power, Ambient_Temp]
        
        # State transition matrix A (3x3)
        self.A = np.array([
            [1.0, 0.0, 0.0],  # SOC dynamics
            [0.0, 0.95, 0.0], # Temperature dynamics (cooling)
            [0.0, 0.0, 0.98]  # Voltage dynamics
        ])
        
        # Control input matrix B (3x2)
        self.B = np.array([
            [1.0/self.battery_capacity, 0.0],  # SOC from charging current
            [self.thermal_resistance, 0.0],    # Temp from current^2 * R
            [0.02, 0.0]  # Voltage response to current
        ])
        
        # Disturbance matrix D (3x3)
        self.D = np.array([
            [0.0, 0.0, 0.0],  # SOC not directly affected by disturbances
            [0.0, 0.0, -0.1], # Temperature affected by ambient
            [0.0, 0.0, 0.0]   # Voltage not directly affected
        ])
    
    def formulate_objective_function(self, states: np.ndarray, controls: np.ndarray, 
                                   predictions: PredictionData) -> float:
        """
        Formulate MPC objective function with adaptive weights
        
        J = w1*economic_cost + w2*soc_deviation + w3*switching_penalty + 
            w4*grid_usage + w5*safety_margin
        """
        horizon = len(controls)
        total_cost = 0.0
        
        # Get current weights (adaptive or fixed)
        if self.enable_adaptive_weights and self.weight_tuner:
            weights = self.weight_tuner.get_weights()
            w1 = weights['economic']
            w2 = weights['soc_deviation']
            w3 = weights['switching']
            w4 = weights['grid_usage']
            w5 = weights['safety']
        else:
            # Use fixed weights from parameters
            w1 = self.params.economic_weight
            w2 = self.params.soc_deviation_weight
            w3 = self.params.switching_penalty_weight
            w4 = self.params.grid_usage_penalty_weight
            w5 = self.params.safety_margin_weight
        
        for k in range(horizon):
            # Extract variables
            soc_k = states[k, 0]
            temp_k = states[k, 1]
            current_k = controls[k, 0] if controls.ndim > 1 else controls[k]
            
            # 1. Economic cost: electricity price * grid power consumption
            grid_power_k = max(0, predictions.load_forecast[k] - predictions.pv_forecast[k])
            economic_cost = w1 * predictions.price_forecast[k] * grid_power_k / 1000.0  # $/hour
            
            # 2. SOC deviation: penalty for deviation from target SOC (70%)
            soc_target = 70.0
            soc_deviation = w2 * (soc_k - soc_target) ** 2
            
            # 3. Switching penalty: penalize frequent switching
            switching_penalty = 0.0
            if k > 0:
                # Compare switching modes between time steps
                prev_mode = 1 if controls[k-1, 1] > 0.5 else 0 if controls.ndim > 1 else 0
                curr_mode = 1 if controls[k, 1] > 0.5 else 0 if controls.ndim > 1 else 0
                if prev_mode != curr_mode:
                    switching_penalty = w3 * 10.0  # Fixed penalty per switch
            
            # 4. Grid usage penalty: minimize grid dependency
            grid_usage_penalty = w4 * (grid_power_k / 1000.0) ** 2
            
            # 5. Safety margin: penalty for approaching safety limits
            safety_penalty = 0.0
            
            # Temperature safety margin
            if temp_k > 40.0:
                safety_penalty += w5 * (temp_k - 40.0) ** 2
            
            # SOC safety margins
            if soc_k < 25.0:
                safety_penalty += w5 * (25.0 - soc_k) ** 2
            elif soc_k > 85.0:
                safety_penalty += w5 * (soc_k - 85.0) ** 2
            
            # Current safety margin
            if abs(current_k) > 25.0:
                safety_penalty += w5 * (abs(current_k) - 25.0) ** 2
            
            # Sum all cost components
            total_cost += economic_cost + soc_deviation + switching_penalty + grid_usage_penalty + safety_penalty
        
        return total_cost
    
    def setup_constraints(self, horizon: int) -> Dict:
        """Setup MPC constraints"""
        constraints = {
            'soc_bounds': (self.params.soc_min, self.params.soc_max),
            'current_bounds': (-self.params.current_max, self.params.current_max),
            'temp_bounds': (0.0, self.params.temp_max),
            'voltage_bounds': (44.0, 56.0),
            'switching_freq': self.params.switching_max_per_hour
        }
        
        return constraints
    
    def predict_states(self, initial_state: np.ndarray, controls: np.ndarray, 
                      predictions: PredictionData) -> np.ndarray:
        """Predict future states using state-space model"""
        horizon = len(controls)
        num_states = len(initial_state)
        
        # Initialize state trajectory
        states = np.zeros((horizon + 1, num_states))
        states[0] = initial_state
        
        for k in range(horizon):
            # Extract control inputs
            if controls.ndim > 1:
                u_k = controls[k]
            else:
                u_k = np.array([controls[k], 0.0])  # Add dummy switching mode
            
            # Extract disturbances
            d_k = np.array([
                predictions.pv_forecast[k] if k < len(predictions.pv_forecast) else 0,
                predictions.load_forecast[k] if k < len(predictions.load_forecast) else 1000,
                predictions.temperature_forecast[k] if k < len(predictions.temperature_forecast) else 25
            ])
            
            # State update: x(k+1) = A*x(k) + B*u(k) + D*d(k)
            states[k+1] = (self.A @ states[k] + 
                          self.B @ u_k + 
                          self.D @ d_k)
            
            # Apply state bounds for stability
            states[k+1, 0] = np.clip(states[k+1, 0], 0, 100)  # SOC bounds
            states[k+1, 1] = np.clip(states[k+1, 1], 0, 60)   # Temperature bounds
            states[k+1, 2] = np.clip(states[k+1, 2], 40, 60)  # Voltage bounds
        
        return states[1:]  # Return predicted states (exclude initial)
    
    def optimize_cvxpy(self, initial_state: SystemState, predictions: PredictionData) -> MPCResult:
        """Optimize using CVXPY solver"""
        start_time = time.time()
        
        try:
            horizon = self.params.horizon
            
            # Decision variables
            current = cp.Variable(horizon)  # Charging current
            switching = cp.Variable(horizon, boolean=True)  # Switching mode (0=PV, 1=Grid)
            
            # State variables (predicted)
            soc = cp.Variable(horizon + 1)
            temp = cp.Variable(horizon + 1)
            voltage = cp.Variable(horizon + 1)
            
            # Constraints list
            constraints = []
            
            # Initial state constraints
            constraints += [
                soc[0] == initial_state.soc,
                temp[0] == initial_state.battery_temp,
                voltage[0] == initial_state.battery_voltage
            ]
            
            # State evolution constraints
            for k in range(horizon):
                # SOC dynamics: SOC(k+1) = SOC(k) + (I_charge * dt * efficiency) / capacity
                constraints += [
                    soc[k+1] == soc[k] + (current[k] * self.params.dt * self.charging_efficiency) / self.battery_capacity
                ]
                
                # Temperature dynamics (simplified)
                ambient_temp = predictions.temperature_forecast[k] if k < len(predictions.temperature_forecast) else 25.0
                constraints += [
                    temp[k+1] == 0.95 * temp[k] + 0.1 * cp.square(current[k]) + 0.05 * ambient_temp
                ]
                
                # Voltage dynamics (simplified)
                constraints += [
                    voltage[k+1] == 0.98 * voltage[k] + 0.02 * current[k]
                ]
            
            # State bounds
            constraints += [
                soc >= self.params.soc_min,
                soc <= self.params.soc_max,
                temp <= self.params.temp_max,
                voltage >= 44.0,
                voltage <= 56.0
            ]
            
            # Control bounds
            constraints += [
                current >= -self.params.current_max,
                current <= self.params.current_max
            ]
            
            # Switching frequency constraint (simplified)
            if horizon > 1:
                constraints += [
                    cp.sum(cp.abs(cp.diff(switching.astype(int)))) <= self.params.switching_max_per_hour
                ]
            
            # Objective function
            cost = 0
            w1, w2, w3, w4, w5 = (self.params.economic_weight, self.params.soc_deviation_weight,
                                 self.params.switching_penalty_weight, self.params.grid_usage_penalty_weight,
                                 self.params.safety_margin_weight)
            
            for k in range(horizon):
                # Economic cost
                grid_power = cp.maximum(0, predictions.load_forecast[k] - predictions.pv_forecast[k])
                price = predictions.price_forecast[k] if k < len(predictions.price_forecast) else 0.25
                cost += w1 * price * grid_power / 1000.0
                
                # SOC deviation
                cost += w2 * cp.square(soc[k+1] - 70.0)
                
                # Grid usage penalty
                cost += w4 * cp.square(grid_power / 1000.0)
                
                # Safety margins
                cost += w5 * cp.maximum(0, temp[k+1] - 40.0) ** 2
                cost += w5 * cp.maximum(0, 25.0 - soc[k+1]) ** 2
                cost += w5 * cp.maximum(0, soc[k+1] - 85.0) ** 2
            
            # Switching penalty
            if horizon > 1:
                cost += w3 * cp.sum(cp.abs(cp.diff(switching.astype(int)))) * 10.0
            
            # Setup and solve problem
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            # Extract results
            solver_time = time.time() - start_time
            
            if problem.status == cp.OPTIMAL:
                optimal_controls = np.column_stack([current.value, switching.value])
                predicted_states = np.column_stack([soc.value[1:], temp.value[1:], voltage.value[1:]])
                
                return MPCResult(
                    success=True,
                    optimal_controls=optimal_controls,
                    predicted_states=predicted_states,
                    objective_value=problem.value,
                    solver_time=solver_time,
                    solver_status=problem.status,
                    economic_cost=problem.value * w1 / (w1 + w2 + w3 + w4 + w5),
                    constraints_satisfied=True
                )
            else:
                logger.warning(f"CVXPY solver failed with status: {problem.status}")
                return self._fallback_solution(initial_state, predictions, solver_time)
                
        except Exception as e:
            logger.error(f"CVXPY optimization error: {e}")
            solver_time = time.time() - start_time
            return self._fallback_solution(initial_state, predictions, solver_time)
    
    def optimize_scipy(self, initial_state: SystemState, predictions: PredictionData) -> MPCResult:
        """Optimize using scipy.optimize (fallback method)"""
        start_time = time.time()
        
        try:
            horizon = self.params.horizon
            
            # Decision variables: [current_0, current_1, ..., switch_0, switch_1, ...]
            n_vars = 2 * horizon
            
            # Initial guess
            x0 = np.zeros(n_vars)
            x0[:horizon] = 5.0  # Initial charging current guess
            x0[horizon:] = 0.0  # Initial switching mode (PV)
            
            # Bounds
            bounds = []
            # Current bounds
            for _ in range(horizon):
                bounds.append((-self.params.current_max, self.params.current_max))
            # Switching bounds (0=PV, 1=Grid)
            for _ in range(horizon):
                bounds.append((0, 1))
            
            bounds = Bounds([b[0] for b in bounds], [b[1] for b in bounds])
            
            # Objective function
            def objective(x):
                currents = x[:horizon]
                switching = x[horizon:]
                controls = np.column_stack([currents, switching])
                
                # Predict states
                initial_state_vec = np.array([initial_state.soc, initial_state.battery_temp, initial_state.battery_voltage])
                states = self.predict_states(initial_state_vec, controls, predictions)
                
                return self.formulate_objective_function(states, controls, predictions)
            
            # Constraints
            def constraint_soc(x):
                currents = x[:horizon]
                switching = x[horizon:]
                controls = np.column_stack([currents, switching])
                
                initial_state_vec = np.array([initial_state.soc, initial_state.battery_temp, initial_state.battery_voltage])
                states = self.predict_states(initial_state_vec, controls, predictions)
                
                # SOC bounds
                soc_violations = []
                for k in range(len(states)):
                    soc_violations.append(states[k, 0] - self.params.soc_min)  # >= soc_min
                    soc_violations.append(self.params.soc_max - states[k, 0])  # <= soc_max
                
                return np.array(soc_violations)
            
            constraints = [
                {'type': 'ineq', 'fun': constraint_soc}
            ]
            
            # Solve optimization
            result = minimize(
                objective, x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            solver_time = time.time() - start_time
            
            if result.success:
                currents = result.x[:horizon]
                switching = result.x[horizon:]
                optimal_controls = np.column_stack([currents, switching])
                
                # Predict final states
                initial_state_vec = np.array([initial_state.soc, initial_state.battery_temp, initial_state.battery_voltage])
                predicted_states = self.predict_states(initial_state_vec, optimal_controls, predictions)
                
                return MPCResult(
                    success=True,
                    optimal_controls=optimal_controls,
                    predicted_states=predicted_states,
                    objective_value=result.fun,
                    solver_time=solver_time,
                    solver_status="optimal",
                    economic_cost=result.fun * 0.3,  # Estimate
                    constraints_satisfied=True
                )
            else:
                logger.warning(f"Scipy optimization failed: {result.message}")
                return self._fallback_solution(initial_state, predictions, solver_time)
                
        except Exception as e:
            logger.error(f"Scipy optimization error: {e}")
            solver_time = time.time() - start_time
            return self._fallback_solution(initial_state, predictions, solver_time)
    
    def _fallback_solution(self, initial_state: SystemState, predictions: PredictionData, solver_time: float) -> MPCResult:
        """Generate fallback solution when optimization fails"""
        horizon = self.params.horizon
        
        # Simple rule-based fallback
        currents = []
        switching = []
        
        for k in range(horizon):
            # Determine charging current based on SOC
            if initial_state.soc < 30:
                current_k = 15.0  # High charging current
            elif initial_state.soc < 70:
                current_k = 10.0  # Medium charging current
            else:
                current_k = 2.0   # Maintenance charging
            
            # Determine switching based on PV availability
            pv_power = predictions.pv_forecast[k] if k < len(predictions.pv_forecast) else 0
            if pv_power > 1000:  # Sufficient PV power
                switch_k = 0  # Use PV
            else:
                switch_k = 1  # Use Grid
            
            currents.append(current_k)
            switching.append(switch_k)
        
        optimal_controls = np.column_stack([currents, switching])
        
        # Generate dummy predicted states
        predicted_states = np.zeros((horizon, 3))
        for k in range(horizon):
            predicted_states[k] = [
                min(90, initial_state.soc + k * 2),  # SOC gradually increases
                initial_state.battery_temp + k * 0.5,  # Temperature rises slightly
                initial_state.battery_voltage  # Voltage stays constant
            ]
        
        return MPCResult(
            success=False,
            optimal_controls=optimal_controls,
            predicted_states=predicted_states,
            objective_value=1000.0,  # High cost for fallback
            solver_time=solver_time,
            solver_status="fallback",
            economic_cost=500.0,
            constraints_satisfied=False
        )
    
    def optimize_hourly(self, initial_state: SystemState, predictions: PredictionData) -> MPCResult:
        """Main optimization function"""
        logger.info(f"Starting MPC optimization - SOC: {initial_state.soc:.1f}%, Solver: {self.solver_type}")
        
        # Choose solver
        if self.solver_type == "cvxpy" and CVXPY_AVAILABLE:
            result = self.optimize_cvxpy(initial_state, predictions)
        elif SCIPY_AVAILABLE:
            result = self.optimize_scipy(initial_state, predictions)
        else:
            logger.error("No optimization solver available!")
            result = self._fallback_solution(initial_state, predictions, 0.001)
        
        # Update statistics
        self.total_optimizations += 1
        self.total_solver_time += result.solver_time
        self.optimization_history.append(result)
        
        logger.info(f"MPC optimization completed - Success: {result.success}, "
                   f"Time: {result.solver_time:.3f}s, Cost: {result.objective_value:.2f}")
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get MPC performance statistics"""
        if not self.optimization_history:
            return {}
        
        recent_results = self.optimization_history[-10:]  # Last 10 optimizations
        
        success_rate = sum(1 for r in recent_results if r.success) / len(recent_results) * 100
        avg_solver_time = np.mean([r.solver_time for r in recent_results])
        avg_objective = np.mean([r.objective_value for r in recent_results])
        
        return {
            'total_optimizations': self.total_optimizations,
            'success_rate_percent': success_rate,
            'avg_solver_time_s': avg_solver_time,
            'avg_objective_value': avg_objective,
            'total_solver_time_s': self.total_solver_time,
            'solver_type': self.solver_type
        }
    
    def export_optimization_data(self, filename: str):
        """Export optimization history"""
        data = {
            'metadata': {
                'solver_type': self.solver_type,
                'total_optimizations': self.total_optimizations,
                'export_timestamp': time.time()
            },
            'parameters': {
                'horizon': self.params.horizon,
                'control_horizon': self.params.control_horizon,
                'weights': {
                    'economic': self.params.economic_weight,
                    'soc_deviation': self.params.soc_deviation_weight,
                    'switching_penalty': self.params.switching_penalty_weight,
                    'grid_usage': self.params.grid_usage_penalty_weight,
                    'safety_margin': self.params.safety_margin_weight
                }
            },
            'history': [
                {
                    'success': result.success,
                    'objective_value': result.objective_value,
                    'solver_time': result.solver_time,
                    'solver_status': result.solver_status,
                    'economic_cost': result.economic_cost,
                    'constraints_satisfied': result.constraints_satisfied
                }
                for result in self.optimization_history
            ],
            'performance': self.get_performance_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"MPC optimization data exported to {filename}")

def test_mpc_optimizer():
    """Test MPC optimizer"""
    print("🧮 MPC OPTIMIZER TEST")
    print("=" * 40)
    
    # Create MPC optimizer
    params = MPCParameters(
        horizon=6,
        economic_weight=1.0,
        soc_deviation_weight=2.0,
        switching_penalty_weight=0.5
    )
    
    mpc = Layer2MPCOptimizer(params)
    
    # Create test scenario
    initial_state = SystemState(
        soc=50.0,
        battery_voltage=12.8,
        battery_current=5.0,
        battery_temp=25.0,
        pv_power=2000.0,
        load_power=1500.0,
        grid_power=0.0,
        electricity_price=0.25,
        timestamp=time.time()
    )
    
    # Create predictions
    predictions = PredictionData(
        pv_forecast=np.array([2000, 3000, 4000, 3000, 1000, 0]),  # 6-hour PV forecast
        load_forecast=np.array([1500, 1800, 2000, 2200, 2500, 2000]),  # 6-hour load forecast
        price_forecast=np.array([0.25, 0.30, 0.35, 0.40, 0.30, 0.25]),  # Price forecast
        temperature_forecast=np.array([25, 27, 30, 32, 28, 25])  # Temperature forecast
    )
    
    # Run optimization
    print(f"\n🔍 Running MPC optimization...")
    print(f"Initial SOC: {initial_state.soc}%")
    print(f"PV forecast: {predictions.pv_forecast}")
    print(f"Load forecast: {predictions.load_forecast}")
    
    result = mpc.optimize_hourly(initial_state, predictions)
    
    print(f"\n📊 Optimization Results:")
    print(f"Success: {result.success}")
    print(f"Solver time: {result.solver_time:.3f}s")
    print(f"Objective value: {result.objective_value:.2f}")
    print(f"Economic cost: {result.economic_cost:.2f}")
    
    if result.success and result.optimal_controls is not None:
        print(f"\n⚡ Optimal Controls:")
        for i, (current, switch) in enumerate(result.optimal_controls):
            mode = "Grid" if switch > 0.5 else "PV"
            print(f"Hour {i+1}: Current={current:.1f}A, Mode={mode}")
        
        print(f"\n📈 Predicted States:")
        for i, (soc, temp, volt) in enumerate(result.predicted_states):
            print(f"Hour {i+1}: SOC={soc:.1f}%, Temp={temp:.1f}°C, Voltage={volt:.1f}V")
    
    # Performance stats
    stats = mpc.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ MPC optimizer test completed!")

if __name__ == "__main__":
    test_mpc_optimizer()

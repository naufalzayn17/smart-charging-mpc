/*
HARDWARE-FIRST SAFETY IMPLEMENTATION
====================================

ESP32-S3 firmware with hardware-enforced safety limits
Independent of software for critical protection functions

Key Features:
- Hardware watchdog timer (5 second timeout)
- Hardware current limiting via shunt + comparator
- Hardware voltage protection via crowbar circuit
- Hardware temperature shutdown via thermal switch
- Software monitoring as secondary layer

Author: Dzaky Naufal K
Date: July 6, 2025
Version: 2.0 - Hardware Safety Priority
*/

#include <Wire.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <esp_task_wdt.h>

// Hardware safety pins (CRITICAL - Cannot be changed by software)
#define HARDWARE_EMERGENCY_PIN    2   // Hardware emergency input
#define HARDWARE_WATCHDOG_PIN     3   // Hardware watchdog output
#define CURRENT_LIMIT_PIN         4   // Hardware current limiting
#define VOLTAGE_CROWBAR_PIN       5   // Hardware overvoltage protection
#define THERMAL_SHUTDOWN_PIN      6   // Hardware thermal protection

// Control relay pins
#define RELAY_PV_PIN             18   // PV source relay
#define RELAY_GRID_PIN           19   // Grid source relay
#define RELAY_CHARGE_PIN         20   // Charge enable relay

// Sensor pins
#define VOLTAGE_SENSE_PIN        35   // Battery voltage sense
#define CURRENT_SENSE_PIN        36   // Current shunt sense
#define PV_VOLTAGE_PIN           37   // PV voltage sense
#define PV_CURRENT_PIN           38   // PV current sense
#define TEMP_SENSOR_PIN          39   // Temperature sensor

// Hardware safety limits (ABSOLUTE LIMITS - DO NOT MODIFY)
#define HARDWARE_VOLTAGE_MAX     14.8   // Hardware crowbar voltage
#define HARDWARE_CURRENT_MAX     32.0   // Hardware fuse rating
#define HARDWARE_TEMP_MAX        65.0   // Hardware thermal cutoff
#define HARDWARE_WATCHDOG_TIMEOUT 5000  // 5 second hardware timeout

// Communication settings
#define UART_BAUD_RATE          115200
#define COMM_TIMEOUT_MS         100
#define MAX_COMM_ERRORS         10

// Global variables
volatile bool hardware_emergency_active = false;
volatile bool software_emergency_active = false;
volatile unsigned long last_watchdog_kick = 0;
volatile unsigned long last_comm_time = 0;

// Sensor data structure
struct SensorReading {
  float battery_voltage;
  float battery_current;
  float battery_temperature;
  float pv_voltage;
  float pv_current;
  float system_temperature;
  unsigned long timestamp;
  bool valid;
};

// Control command structure
struct ControlCommand {
  float charge_current;
  bool pv_enable;
  bool grid_enable;
  unsigned long timestamp;
  bool emergency_stop;
};

// System state
SensorReading current_sensors;
ControlCommand current_command;
int communication_errors = 0;
bool system_initialized = false;

// Temperature sensor setup
OneWire oneWire(TEMP_SENSOR_PIN);
DallasTemperature temperature_sensor(&oneWire);

// Hardware interrupt service routines
void IRAM_ATTR hardware_emergency_isr() {
  // IMMEDIATE hardware shutdown - no software delays
  digitalWrite(RELAY_PV_PIN, LOW);
  digitalWrite(RELAY_GRID_PIN, LOW);
  digitalWrite(RELAY_CHARGE_PIN, LOW);
  hardware_emergency_active = true;
  
  // Log emergency via serial (if possible)
  Serial.println("{\"emergency\":\"hardware_triggered\",\"timestamp\":" + String(millis()) + "}");
}

void IRAM_ATTR hardware_watchdog_isr() {
  // Hardware watchdog timeout
  if (millis() - last_watchdog_kick > HARDWARE_WATCHDOG_TIMEOUT) {
    hardware_emergency_isr();
  }
}

void setup() {
  // Initialize serial FIRST
  Serial.begin(UART_BAUD_RATE);
  
  // Hardware safety setup BEFORE anything else
  setup_hardware_safety();
  
  // Initialize GPIO pins
  setup_gpio_pins();
  
  // Initialize sensors
  setup_sensors();
  
  // Setup hardware watchdog
  setup_hardware_watchdog();
  
  // Enable hardware interrupts
  setup_hardware_interrupts();
  
  // Initial safety check
  if (perform_initial_safety_check()) {
    system_initialized = true;
    Serial.println("{\"status\":\"initialized\",\"timestamp\":" + String(millis()) + "}");
  } else {
    Serial.println("{\"error\":\"initialization_failed\",\"timestamp\":" + String(millis()) + "}");
    hardware_emergency_isr();
  }
}

void setup_hardware_safety() {
  // Hardware emergency input (normally high, emergency on low)
  pinMode(HARDWARE_EMERGENCY_PIN, INPUT_PULLUP);
  
  // Hardware watchdog output
  pinMode(HARDWARE_WATCHDOG_PIN, OUTPUT);
  digitalWrite(HARDWARE_WATCHDOG_PIN, HIGH);
  
  // Hardware current limiting (comparator output)
  pinMode(CURRENT_LIMIT_PIN, INPUT);
  
  // Hardware voltage crowbar trigger
  pinMode(VOLTAGE_CROWBAR_PIN, OUTPUT);
  digitalWrite(VOLTAGE_CROWBAR_PIN, LOW);  // Crowbar disabled initially
  
  // Hardware thermal shutdown input
  pinMode(THERMAL_SHUTDOWN_PIN, INPUT_PULLUP);
  
  Serial.println("{\"info\":\"hardware_safety_initialized\"}");
}

void setup_gpio_pins() {
  // Control relay pins (all OFF initially)
  pinMode(RELAY_PV_PIN, OUTPUT);
  pinMode(RELAY_GRID_PIN, OUTPUT);
  pinMode(RELAY_CHARGE_PIN, OUTPUT);
  
  digitalWrite(RELAY_PV_PIN, LOW);
  digitalWrite(RELAY_GRID_PIN, LOW);
  digitalWrite(RELAY_CHARGE_PIN, LOW);
  
  // Sensor pins
  pinMode(VOLTAGE_SENSE_PIN, INPUT);
  pinMode(CURRENT_SENSE_PIN, INPUT);
  pinMode(PV_VOLTAGE_PIN, INPUT);
  pinMode(PV_CURRENT_PIN, INPUT);
}

void setup_sensors() {
  // Initialize temperature sensor
  temperature_sensor.begin();
  temperature_sensor.setResolution(10);  // 10-bit resolution for faster conversion
  
  // Initialize ADC
  analogSetAttenuation(ADC_11db);  // For 0-3.3V input
  analogReadResolution(12);        // 12-bit resolution
}

void setup_hardware_watchdog() {
  // ESP32 hardware watchdog
  esp_task_wdt_init(HARDWARE_WATCHDOG_TIMEOUT / 1000, true);
  esp_task_wdt_add(NULL);
  
  last_watchdog_kick = millis();
}

void setup_hardware_interrupts() {
  // Hardware emergency interrupt
  attachInterrupt(digitalPinToInterrupt(HARDWARE_EMERGENCY_PIN), 
                  hardware_emergency_isr, FALLING);
  
  // Current limit interrupt
  attachInterrupt(digitalPinToInterrupt(CURRENT_LIMIT_PIN),
                  hardware_emergency_isr, RISING);
  
  // Thermal shutdown interrupt
  attachInterrupt(digitalPinToInterrupt(THERMAL_SHUTDOWN_PIN),
                  hardware_emergency_isr, FALLING);
}

bool perform_initial_safety_check() {
  // Read initial sensor values
  read_sensors_blocking();
  
  // Check if sensors are within safe ranges
  if (current_sensors.battery_voltage > HARDWARE_VOLTAGE_MAX ||
      current_sensors.battery_voltage < 8.0 ||
      current_sensors.battery_temperature > HARDWARE_TEMP_MAX ||
      current_sensors.battery_temperature < -10.0) {
    return false;
  }
  
  return true;
}

void loop() {
  // Kick hardware watchdog FIRST
  kick_hardware_watchdog();
  
  // Check hardware safety conditions
  if (!check_hardware_safety()) {
    return;  // Emergency active, do nothing
  }
  
  // Read sensors (non-blocking when possible)
  read_sensors_non_blocking();
  
  // Process communication
  process_serial_communication();
  
  // Execute control commands (if safe)
  execute_control_commands();
  
  // Monitor communication timeout
  check_communication_timeout();
  
  // Small delay to prevent watchdog issues
  delay(10);
}

void kick_hardware_watchdog() {
  // Kick ESP32 software watchdog
  esp_task_wdt_reset();
  
  // Kick external hardware watchdog
  digitalWrite(HARDWARE_WATCHDOG_PIN, LOW);
  delayMicroseconds(100);
  digitalWrite(HARDWARE_WATCHDOG_PIN, HIGH);
  
  last_watchdog_kick = millis();
}

bool check_hardware_safety() {
  // Check hardware emergency pin
  if (digitalRead(HARDWARE_EMERGENCY_PIN) == LOW) {
    if (!hardware_emergency_active) {
      Serial.println("{\"error\":\"hardware_emergency_pin\",\"timestamp\":" + String(millis()) + "}");
      hardware_emergency_isr();
    }
    return false;
  }
  
  // Check current limit pin
  if (digitalRead(CURRENT_LIMIT_PIN) == HIGH) {
    if (!hardware_emergency_active) {
      Serial.println("{\"error\":\"hardware_current_limit\",\"timestamp\":" + String(millis()) + "}");
      hardware_emergency_isr();
    }
    return false;
  }
  
  // Check thermal shutdown pin
  if (digitalRead(THERMAL_SHUTDOWN_PIN) == LOW) {
    if (!hardware_emergency_active) {
      Serial.println("{\"error\":\"hardware_thermal_shutdown\",\"timestamp\":" + String(millis()) + "}");
      hardware_emergency_isr();
    }
    return false;
  }
  
  // Check software-measured values
  if (current_sensors.valid) {
    // Voltage protection
    if (current_sensors.battery_voltage > HARDWARE_VOLTAGE_MAX) {
      trigger_voltage_crowbar();
      return false;
    }
    
    // Current protection (software backup)
    if (abs(current_sensors.battery_current) > HARDWARE_CURRENT_MAX) {
      if (!software_emergency_active) {
        Serial.println("{\"error\":\"software_current_limit\",\"current\":" + String(current_sensors.battery_current) + "}");
        software_emergency_stop();
      }
      return false;
    }
    
    // Temperature protection (software backup)
    if (current_sensors.battery_temperature > HARDWARE_TEMP_MAX) {
      if (!software_emergency_active) {
        Serial.println("{\"error\":\"software_temperature_limit\",\"temperature\":" + String(current_sensors.battery_temperature) + "}");
        software_emergency_stop();
      }
      return false;
    }
  }
  
  return true;
}

void trigger_voltage_crowbar() {
  // Trigger hardware crowbar circuit
  digitalWrite(VOLTAGE_CROWBAR_PIN, HIGH);
  
  Serial.println("{\"emergency\":\"voltage_crowbar_triggered\",\"voltage\":" + String(current_sensors.battery_voltage) + "}");
  
  // Keep crowbar active for 1 second
  delay(1000);
  digitalWrite(VOLTAGE_CROWBAR_PIN, LOW);
  
  hardware_emergency_isr();
}

void software_emergency_stop() {
  software_emergency_active = true;
  
  // Turn off all relays
  digitalWrite(RELAY_PV_PIN, LOW);
  digitalWrite(RELAY_GRID_PIN, LOW);
  digitalWrite(RELAY_CHARGE_PIN, LOW);
  
  Serial.println("{\"emergency\":\"software_triggered\",\"timestamp\":" + String(millis()) + "}");
}

void read_sensors_blocking() {
  // Voltage measurements (with calibration)
  int voltage_raw = analogRead(VOLTAGE_SENSE_PIN);
  current_sensors.battery_voltage = (voltage_raw / 4095.0) * 3.3 * 5.0;  // Voltage divider
  
  int pv_voltage_raw = analogRead(PV_VOLTAGE_PIN);
  current_sensors.pv_voltage = (pv_voltage_raw / 4095.0) * 3.3 * 15.0;  // Higher voltage divider
  
  // Current measurements (with shunt calibration)
  int current_raw = analogRead(CURRENT_SENSE_PIN);
  float current_voltage = (current_raw / 4095.0) * 3.3;
  current_sensors.battery_current = (current_voltage - 1.65) / 0.1;  // 100mV/A shunt
  
  int pv_current_raw = analogRead(PV_CURRENT_PIN);
  float pv_current_voltage = (pv_current_raw / 4095.0) * 3.3;
  current_sensors.pv_current = (pv_current_voltage - 1.65) / 0.1;
  
  // Temperature measurement (blocking)
  temperature_sensor.requestTemperatures();
  current_sensors.battery_temperature = temperature_sensor.getTempCByIndex(0);
  
  // System temperature (internal sensor)
  current_sensors.system_temperature = temperatureRead();
  
  current_sensors.timestamp = millis();
  current_sensors.valid = true;
}

void read_sensors_non_blocking() {
  static unsigned long last_sensor_read = 0;
  static bool temp_conversion_started = false;
  static unsigned long temp_conversion_time = 0;
  
  unsigned long current_time = millis();
  
  // Read voltage and current every 50ms (20Hz)
  if (current_time - last_sensor_read >= 50) {
    // Quick voltage and current readings
    int voltage_raw = analogRead(VOLTAGE_SENSE_PIN);
    current_sensors.battery_voltage = (voltage_raw / 4095.0) * 3.3 * 5.0;
    
    int current_raw = analogRead(CURRENT_SENSE_PIN);
    float current_voltage = (current_raw / 4095.0) * 3.3;
    current_sensors.battery_current = (current_voltage - 1.65) / 0.1;
    
    int pv_voltage_raw = analogRead(PV_VOLTAGE_PIN);
    current_sensors.pv_voltage = (pv_voltage_raw / 4095.0) * 3.3 * 15.0;
    
    int pv_current_raw = analogRead(PV_CURRENT_PIN);
    float pv_current_voltage = (pv_current_raw / 4095.0) * 3.3;
    current_sensors.pv_current = (pv_current_voltage - 1.65) / 0.1;
    
    current_sensors.system_temperature = temperatureRead();
    
    last_sensor_read = current_time;
  }
  
  // Non-blocking temperature reading (slower update rate)
  if (!temp_conversion_started) {
    temperature_sensor.requestTemperatures();
    temp_conversion_started = true;
    temp_conversion_time = current_time;
  } else if (current_time - temp_conversion_time >= 100) {  // 100ms conversion time
    current_sensors.battery_temperature = temperature_sensor.getTempCByIndex(0);
    temp_conversion_started = false;
  }
  
  current_sensors.timestamp = current_time;
  current_sensors.valid = true;
}

void process_serial_communication() {
  if (Serial.available()) {
    String incoming_data = Serial.readStringUntil('\n');
    incoming_data.trim();
    
    if (incoming_data.length() > 0) {
      last_comm_time = millis();
      communication_errors = 0;  // Reset error count on successful communication
      
      process_command(incoming_data);
    }
  }
}

void process_command(String command_string) {
  // Parse JSON command
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, command_string);
  
  if (error) {
    Serial.println("{\"error\":\"json_parse_failed\",\"data\":\"" + command_string + "\"}");
    return;
  }
  
  String action = doc["action"];
  
  if (action == "read_sensors") {
    send_sensor_data();
  } else if (action == "control") {
    process_control_command(doc);
  } else if (action == "emergency_stop") {
    software_emergency_stop();
    Serial.println("{\"response\":\"emergency_stop_executed\"}");
  } else if (action == "reset_emergency") {
    reset_emergency_state();
  } else if (action == "status") {
    send_status_data();
  } else {
    Serial.println("{\"error\":\"unknown_command\",\"action\":\"" + action + "\"}");
  }
}

void send_sensor_data() {
  DynamicJsonDocument doc(1024);
  
  doc["timestamp"] = current_sensors.timestamp;
  doc["battery_voltage"] = current_sensors.battery_voltage;
  doc["battery_current"] = current_sensors.battery_current;
  doc["battery_temp"] = current_sensors.battery_temperature;
  doc["pv_voltage"] = current_sensors.pv_voltage;
  doc["pv_current"] = current_sensors.pv_current;
  doc["system_temp"] = current_sensors.system_temperature;
  doc["valid"] = current_sensors.valid;
  doc["emergency_active"] = hardware_emergency_active || software_emergency_active;
  
  String response;
  serializeJson(doc, response);
  Serial.println(response);
}

void process_control_command(DynamicJsonDocument& doc) {
  // Extract control parameters
  current_command.charge_current = doc["charge_current"];
  current_command.pv_enable = doc["pv_enable"];
  current_command.grid_enable = doc["grid_enable"];
  current_command.timestamp = doc["timestamp"];
  current_command.emergency_stop = doc.containsKey("emergency_stop") ? doc["emergency_stop"] : false;
  
  // Validate control command
  if (validate_control_command()) {
    // Execute control if safe
    if (execute_control_safely()) {
      Serial.println("ACK");
    } else {
      Serial.println("{\"error\":\"control_execution_failed\"}");
    }
  } else {
    Serial.println("{\"error\":\"control_command_invalid\"}");
  }
}

bool validate_control_command() {
  // Check for emergency stop
  if (current_command.emergency_stop) {
    return true;  // Always allow emergency stop
  }
  
  // Check if system is in emergency state
  if (hardware_emergency_active || software_emergency_active) {
    return false;  // No control allowed during emergency
  }
  
  // Validate charge current
  if (current_command.charge_current < 0 || current_command.charge_current > HARDWARE_CURRENT_MAX) {
    return false;
  }
  
  // Check current sensor readings
  if (!current_sensors.valid) {
    return false;
  }
  
  // Voltage-based validation
  if (current_sensors.battery_voltage > 14.0 && current_command.charge_current > 5.0) {
    return false;  // Limit charging at high voltage
  }
  
  if (current_sensors.battery_voltage < 10.5 && current_command.charge_current > 20.0) {
    return false;  // Limit charging at very low voltage
  }
  
  // Temperature-based validation
  if (current_sensors.battery_temperature > 50.0 && current_command.charge_current > 10.0) {
    return false;  // Reduce charging at high temperature
  }
  
  return true;
}

bool execute_control_safely() {
  if (current_command.emergency_stop) {
    software_emergency_stop();
    return true;
  }
  
  // Final safety check before execution
  if (!check_hardware_safety()) {
    return false;
  }
  
  // Execute relay control
  digitalWrite(RELAY_PV_PIN, current_command.pv_enable ? HIGH : LOW);
  digitalWrite(RELAY_GRID_PIN, current_command.grid_enable ? HIGH : LOW);
  
  // Charge relay control based on current command
  if (current_command.charge_current > 0.5) {
    digitalWrite(RELAY_CHARGE_PIN, HIGH);
  } else {
    digitalWrite(RELAY_CHARGE_PIN, LOW);
  }
  
  return true;
}

void execute_control_commands() {
  // This function handles any deferred control execution
  // Currently, control is executed immediately in process_control_command
  // This function can be used for PWM control or other time-based operations
  
  static unsigned long last_control_update = 0;
  unsigned long current_time = millis();
  
  // Update control outputs every 100ms
  if (current_time - last_control_update >= 100) {
    // Check if current command is still valid
    if (current_time - current_command.timestamp > 10000) {  // 10 second timeout
      // Command timeout - go to safe state
      current_command.charge_current = 0;
      current_command.pv_enable = false;
      current_command.grid_enable = false;
      
      digitalWrite(RELAY_PV_PIN, LOW);
      digitalWrite(RELAY_GRID_PIN, LOW);
      digitalWrite(RELAY_CHARGE_PIN, LOW);
    }
    
    last_control_update = current_time;
  }
}

void check_communication_timeout() {
  unsigned long current_time = millis();
  
  // Check for communication timeout
  if (current_time - last_comm_time > 30000) {  // 30 second timeout
    communication_errors++;
    
    if (communication_errors > MAX_COMM_ERRORS) {
      Serial.println("{\"error\":\"communication_timeout\",\"errors\":" + String(communication_errors) + "}");
      
      // Go to safe state on communication loss
      current_command.charge_current = 0;
      current_command.pv_enable = false;
      current_command.grid_enable = false;
      
      digitalWrite(RELAY_PV_PIN, LOW);
      digitalWrite(RELAY_GRID_PIN, LOW);
      digitalWrite(RELAY_CHARGE_PIN, LOW);
    }
    
    last_comm_time = current_time;  // Reset to prevent spam
  }
}

void reset_emergency_state() {
  // Only allow reset if hardware emergency is not active
  if (digitalRead(HARDWARE_EMERGENCY_PIN) == HIGH &&
      digitalRead(CURRENT_LIMIT_PIN) == LOW &&
      digitalRead(THERMAL_SHUTDOWN_PIN) == HIGH) {
    
    hardware_emergency_active = false;
    software_emergency_active = false;
    
    Serial.println("{\"response\":\"emergency_reset\",\"timestamp\":" + String(millis()) + "}");
  } else {
    Serial.println("{\"error\":\"hardware_emergency_still_active\"}");
  }
}

void send_status_data() {
  DynamicJsonDocument doc(1024);
  
  doc["timestamp"] = millis();
  doc["system_initialized"] = system_initialized;
  doc["hardware_emergency"] = hardware_emergency_active;
  doc["software_emergency"] = software_emergency_active;
  doc["communication_errors"] = communication_errors;
  doc["last_comm_time"] = last_comm_time;
  doc["uptime"] = millis();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["cpu_frequency"] = ESP.getCpuFreqMHz();
  
  // Hardware status
  doc["emergency_pin"] = digitalRead(HARDWARE_EMERGENCY_PIN);
  doc["current_limit_pin"] = digitalRead(CURRENT_LIMIT_PIN);
  doc["thermal_pin"] = digitalRead(THERMAL_SHUTDOWN_PIN);
  
  // Control status
  doc["pv_relay"] = digitalRead(RELAY_PV_PIN);
  doc["grid_relay"] = digitalRead(RELAY_GRID_PIN);
  doc["charge_relay"] = digitalRead(RELAY_CHARGE_PIN);
  
  String response;
  serializeJson(doc, response);
  Serial.println(response);
}

// IR_Sensor_Read.ino
// Reads IR break-beam sensor for basketball make detection
// For Raspberry Pi GPIO or Arduino

#define IR_SENSOR_PIN 2  // GPIO pin for IR sensor
#define DEBOUNCE_TIME 100  // ms to ignore after trigger

unsigned long lastTriggerTime = 0;
bool shotDetected = false;

void setup() {
  pinMode(IR_SENSOR_PIN, INPUT_PULLUP);
  Serial.begin(9600);
  attachInterrupt(digitalPinToInterrupt(IR_SENSOR_PIN), 
                  onSensorTrigger, FALLING);
}

void loop() {
  if (shotDetected) {
    Serial.println("MAKE_DETECTED");
    sendToBackend();  // Send to processing subsystem
    shotDetected = false;
  }
}

void onSensorTrigger() {
  unsigned long currentTime = millis();
  
  // Debouncing: ignore triggers within 100ms
  if (currentTime - lastTriggerTime > DEBOUNCE_TIME) {
    shotDetected = true;
    lastTriggerTime = currentTime;
  }
}

void sendToBackend() {
  // Send JSON via serial/Bluetooth
  Serial.println("{\"event\":\"make\",\"timestamp\":" + 
                 String(millis()) + "}");
}

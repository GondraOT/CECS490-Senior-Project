#include <Arduino.h>

// ================= CONFIG =================
const int IR_PIN = 27;

const int PIEZO_PINS[] = {32, 35, 34};
const int NUM_PIEZO = sizeof(PIEZO_PINS) / sizeof(PIEZO_PINS[0]);

const uint32_t SHOT_WINDOW_MS    = 1200;  // 1.2s make window
const uint32_t SWISH_WINDOW_MS   = 1000;
const uint32_t GLOBAL_COOLDOWN_MS = 300;

const int     HIT_THRESHOLD    = 150;
const uint32_t PIEZO_REFRACT_MS = 120;

// ================= STATE =================
void IRAM_ATTR beamISR();

volatile bool irChanged = false;
volatile bool irBroken  = false;

float    baseline[3];
uint32_t lastHitMs[3];
uint32_t lastAnyPiezoHitMs = 0;

enum ShotState { IDLE, WAIT_FOR_IR, COOLDOWN };
ShotState state = IDLE;

uint32_t windowStartMs  = 0;
bool     madeThisWindow = false;

// ================= ISR =================
void IRAM_ATTR beamISR() {
  irBroken  = (digitalRead(IR_PIN) == LOW);
  irChanged = true;
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  pinMode(IR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IR_PIN), beamISR, CHANGE);

  analogReadResolution(12);

  for (int i = 0; i < NUM_PIEZO; i++) {
    baseline[i]  = analogRead(PIEZO_PINS[i]);
    lastHitMs[i] = 0;
  }

  Serial.println("=== Hoop IQ Ready ===");
}

// ================= LOOP =================
void loop() {
  uint32_t nowMs = millis();

  // ---------- IR EVENTS ----------
  if (irChanged) {
    noInterrupts();
    bool broken   = irBroken;
    irChanged     = false;
    interrupts();

    if (broken) {
      if (state == WAIT_FOR_IR) {
        // Ball broke the IR beam during the shot window = MAKE
        madeThisWindow = true;
      }
      else if (state == IDLE && (nowMs - lastAnyPiezoHitMs) > SWISH_WINDOW_MS) {
        // IR broken with no prior piezo hit = swish
        Serial.println("SWISH:1");   // Rust listens for MAKE:
        state          = COOLDOWN;
        windowStartMs  = nowMs;
      }
    }
  }

  // ---------- PIEZO DETECTION ----------
  bool hitDetected = false;

  for (int i = 0; i < NUM_PIEZO; i++) {
    int raw   = analogRead(PIEZO_PINS[i]);
    int delta = abs(raw - (int)baseline[i]);

    // Slow baseline tracking
    baseline[i] = baseline[i] * 0.998f + raw * 0.002f;

    if (delta > HIT_THRESHOLD && (nowMs - lastHitMs[i]) > PIEZO_REFRACT_MS) {
      lastHitMs[i] = nowMs;
      hitDetected  = true;
    }
  }

  // ---------- FIRST HIT STARTS TIMER ----------
  if (hitDetected) {
    lastAnyPiezoHitMs = nowMs;

    if (state == IDLE) {
      state          = WAIT_FOR_IR;
      windowStartMs  = nowMs;
      madeThisWindow = false;
      Serial.println("RIM:1");    // Rust listens for RIM: — rim/backboard contact
    }
    // Extra hits within the window are ignored (no timer reset)
  }

  // ---------- RESULT LOGIC ----------
  switch (state) {

    case IDLE:
      break;

    case WAIT_FOR_IR:
      if (nowMs - windowStartMs >= SHOT_WINDOW_MS) {
        if (madeThisWindow)
          Serial.println("MAKE:1");   // Piezo hit + IR broken = made
        else
          Serial.println("BACK:1");   // Piezo hit, no IR = miss (backboard/rim)

        state         = COOLDOWN;
        windowStartMs = nowMs;
      }
      break;

    case COOLDOWN:
      if (nowMs - windowStartMs >= GLOBAL_COOLDOWN_MS) {
        state = IDLE;
      }
      break;
  }
}

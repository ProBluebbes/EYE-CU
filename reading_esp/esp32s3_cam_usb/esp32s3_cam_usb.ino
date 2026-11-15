// ESP32-S3 camera + speaker for Eye-CU, over one USB cable (no WiFi, no Bluetooth).
//
// ESP32 -> laptop (camera):  [FF AA BB CC] [4-byte little-endian length] [JPEG bytes]
// laptop -> ESP32 (voice):   [FF AA BB DD] [type: 'P' audio | 'S' stop | 'T' test beep] [4-byte LE length] [data]
//     'P' data = 16 kHz mono signed 16-bit little-endian PCM, appended to the play queue
//     'S' (length 0) = stop: drop whatever is still queued (a new prompt cuts off the old one)
//     'T' (length 0) = 1 s, 2 kHz test beep (checks the wiring / buzzer without any audio data)
// The voice plays on a passive piezo buzzer wired BETWEEN two pins (push-pull): both carry a 62.5 kHz
// PWM, BUZZER_PIN with duty d and BUZZER_PIN_B with 255 - d, so the piezo sees +-3.3 V (6.6 V
// peak-to-peak, twice a single pin) following each audio sample, 16,000 times a second.
//
// Arduino IDE settings (esp32 core 3.x):
//   Tools > PSRAM            > "OPI PSRAM"
//   Tools > USB CDC On Boot  > "Enabled"   (plug into the native USB port, e.g. the port
//                                           labelled "USB" on Freenove, or the XIAO's port)
//   If you're plugged into a "UART"/"COM" port instead, leave CDC disabled - it still works,
//   just slower (limited by the 2 Mbaud USB-serial chip).
//
// Wiring: piezo leg 1 to BUZZER_PIN, leg 2 to BUZZER_PIN_B (not GND). Optional: 100-220 ohm in
// series with one leg. Needs a passive piezo; an *active* buzzer can only beep.

#include "esp_camera.h"

// ---- Pick your board (uncomment ONE) ----
#define BOARD_ESP32S3_EYE_OR_FREENOVE
// #define BOARD_XIAO_ESP32S3_SENSE

// ---- Speaker pins: two free GPIOs (NOT 4/5 on Freenove: those are the camera's SIOD/SIOC).
#if defined(BOARD_XIAO_ESP32S3_SENSE)
  #define BUZZER_PIN    2   // D1
  #define BUZZER_PIN_B  3   // D2
#else
  #define BUZZER_PIN   14   // Freenove: left header, next to 5V
  #define BUZZER_PIN_B 21   // Freenove: right header, between GPIO47 and GPIO20
#endif
#define BUZZER_LEDC_CH 4    // LEDC channels 4 + 5 share timer 2 (in step); the camera clock uses 0 / timer 0
#define PWM_FREQ   62500    // carrier, above hearing; 8-bit duty


#if defined(BOARD_ESP32S3_EYE_OR_FREENOVE)
  #define PWDN_GPIO_NUM  -1
  #define RESET_GPIO_NUM -1
  #define XCLK_GPIO_NUM  15
  #define SIOD_GPIO_NUM   4
  #define SIOC_GPIO_NUM   5
  #define Y9_GPIO_NUM    16
  #define Y8_GPIO_NUM    17
  #define Y7_GPIO_NUM    18
  #define Y6_GPIO_NUM    12
  #define Y5_GPIO_NUM    10
  #define Y4_GPIO_NUM     8
  #define Y3_GPIO_NUM     9
  #define Y2_GPIO_NUM    11
  #define VSYNC_GPIO_NUM  6
  #define HREF_GPIO_NUM   7
  #define PCLK_GPIO_NUM  13
#elif defined(BOARD_XIAO_ESP32S3_SENSE)
  #define PWDN_GPIO_NUM  -1
  #define RESET_GPIO_NUM -1
  #define XCLK_GPIO_NUM  10
  #define SIOD_GPIO_NUM  40
  #define SIOC_GPIO_NUM  39
  #define Y9_GPIO_NUM    48
  #define Y8_GPIO_NUM    11
  #define Y7_GPIO_NUM    12
  #define Y6_GPIO_NUM    14
  #define Y5_GPIO_NUM    16
  #define Y4_GPIO_NUM    18
  #define Y3_GPIO_NUM    17
  #define Y2_GPIO_NUM    15
  #define VSYNC_GPIO_NUM 38
  #define HREF_GPIO_NUM  47
  #define PCLK_GPIO_NUM  13
#endif

static const uint8_t MAGIC[4]    = {0xFF, 0xAA, 0xBB, 0xCC};  // frame header (ESP32 -> laptop)
static const uint8_t MAGIC_IN[4] = {0xFF, 0xAA, 0xBB, 0xDD};  // audio header (laptop -> ESP32)
static const int SAMPLE_RATE = 16000;
static const size_t QUEUE_BYTES = 1024 * 1024;  // ~32 s of audio, in PSRAM

bool camera_ok = false;
bool speaker_ok = false;
static volatile bool test_beep = false;

// ---------------------------------------------------------------- audio queue (ring buffer)
static uint8_t* q_buf = nullptr;
static size_t q_size = 0;
static volatile size_t q_head = 0, q_tail = 0;   // write at head, read at tail
static portMUX_TYPE q_mux = portMUX_INITIALIZER_UNLOCKED;

static size_t q_used() {
  portENTER_CRITICAL(&q_mux);
  size_t u = (q_head + q_size - q_tail) % q_size;
  portEXIT_CRITICAL(&q_mux);
  return u;
}

static void q_clear() {
  portENTER_CRITICAL(&q_mux);
  q_tail = q_head;
  portEXIT_CRITICAL(&q_mux);
}

static void q_push(uint8_t b) {
  while (q_used() >= q_size - 2) vTaskDelay(1);  // full: wait for playback (USB flow control does the rest)
  q_buf[q_head] = b;
  portENTER_CRITICAL(&q_mux);
  q_head = (q_head + 1) % q_size;
  portEXIT_CRITICAL(&q_mux);
}

static size_t q_pop(uint8_t* out, size_t n) {
  size_t have = q_used();
  if (n > have) n = have;
  for (size_t i = 0; i < n; i++) out[i] = q_buf[(q_tail + i) % q_size];
  portENTER_CRITICAL(&q_mux);
  q_tail = (q_tail + n) % q_size;
  portEXIT_CRITICAL(&q_mux);
  return n;
}

// ---------------------------------------------------------------- laptop -> ESP32 (voice)
static void rx_task(void*) {
  uint8_t win[4] = {0, 0, 0, 0};
  for (;;) {
    if (!Serial.available()) { vTaskDelay(1); continue; }
    // slide a 4-byte window until it matches the audio header
    win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = (uint8_t)Serial.read();
    if (memcmp(win, MAGIC_IN, 4) != 0) continue;
    memset(win, 0, 4);
    uint8_t hdr[5];
    if (Serial.readBytes(hdr, 5) != 5) continue;
    uint8_t type = hdr[0];
    uint32_t len = hdr[1] | (hdr[2] << 8) | (hdr[3] << 16) | ((uint32_t)hdr[4] << 24);
    if (type == 'S') { q_clear(); continue; }
    if (type == 'T') { q_clear(); test_beep = true; continue; }
    if (type != 'P' || len > 4 * 1024 * 1024) continue;  // unknown or corrupt: resync on next header
    uint8_t chunk[512];
    while (len > 0) {
      size_t want = len < sizeof(chunk) ? len : sizeof(chunk);
      size_t got = Serial.readBytes(chunk, want);
      if (got == 0) break;                                // timed out mid-packet: give up on it
      for (size_t i = 0; i < got; i++) q_push(chunk[i]);
      len -= got;
    }
  }
}

static inline void drive(uint32_t d) {   // d = 0..255; 128 = no net voltage on the piezo
  ledcWrite(BUZZER_PIN, d);
  ledcWrite(BUZZER_PIN_B, 255 - d);
}
static inline void pwm_out(int16_t sample) {
  drive((uint32_t)((sample >> 8) + 128));     // signed 16-bit -> 8-bit duty around 50%
}
static inline void quiet_out() {              // both pins low: no voltage, no carrier
  ledcWrite(BUZZER_PIN, 0);
  ledcWrite(BUZZER_PIN_B, 0);
}

// One sample every 62.5 us, timed against the microsecond clock (no drift). While audio is queued
// this task spins; it yields between samples so the idle task (and its watchdog) still runs.
static void play_task(void*) {
  int16_t s;
  for (;;) {
    if (test_beep) {                                    // 2 kHz square wave for 1 s
      test_beep = false;
      int64_t t = esp_timer_get_time();
      for (int i = 0; i < 4000; i++) {
        drive((i & 1) ? 255 : 0);              // full +-3.3 V square
        t += 250;
        while (esp_timer_get_time() < t) { }
        if (i % 64 == 0) taskYIELD();
      }
      quiet_out();
      continue;
    }
    if (q_used() < 2) { quiet_out(); vTaskDelay(2); continue; }  // idle: no voltage on the piezo
    int64_t t = esp_timer_get_time();
    int n = 0;
    while (q_pop((uint8_t*)&s, 2) == 2) {
      pwm_out(s);
      t += 1000000 / SAMPLE_RATE;
      while (esp_timer_get_time() < t) { }
      if (++n % 64 == 0) taskYIELD();
    }
    quiet_out();
  }
}

// ---------------------------------------------------------------- setup / camera loop
void setup() {
  Serial.setRxBufferSize(16384);
  Serial.begin(2000000);  // ignored on native USB (it's always full speed); matters on UART ports
  Serial.setTimeout(500);
  delay(1000);

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.frame_size = FRAMESIZE_VGA;   // 640x480. Drop to FRAMESIZE_QVGA if it's slow.
  config.jpeg_quality = 12;            // higher number = smaller frames = faster
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if (!psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  camera_ok = (esp_camera_init(&config) == ESP_OK);
  // Text on the same port is fine: the laptop skips anything that isn't a frame.
  Serial.println(camera_ok ? "CAMERA_OK" : "CAMERA_INIT_FAILED (wrong board pins?)");

  if (camera_ok) {
    sensor_t* s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) s->set_vflip(s, 1);
    // s->set_hmirror(s, 1);   // uncomment if mirrored
  }

  // speaker: queue in PSRAM (small DRAM one without PSRAM), PWM on BUZZER_PIN
  q_size = psramFound() ? QUEUE_BYTES : 64 * 1024;
  q_buf = (uint8_t*)(psramFound() ? ps_malloc(q_size) : malloc(q_size));
  speaker_ok = q_buf && ledcAttachChannel(BUZZER_PIN, PWM_FREQ, 8, BUZZER_LEDC_CH)
                     && ledcAttachChannel(BUZZER_PIN_B, PWM_FREQ, 8, BUZZER_LEDC_CH + 1);
  if (speaker_ok) quiet_out();
  Serial.println(speaker_ok ? "SPEAKER_OK" : "SPEAKER_INIT_FAILED (check BUZZER_PIN / BUZZER_PIN_B)");
  if (speaker_ok) {
    xTaskCreatePinnedToCore(rx_task, "rx", 4096, nullptr, 3, nullptr, 0);
    xTaskCreatePinnedToCore(play_task, "play", 4096, nullptr, 0, nullptr, 0);  // idle priority: shares core 0 fairly
  }
}

void loop() {
  if (!camera_ok) { delay(1000); return; }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return;

  uint32_t len = fb->len;
  Serial.write(MAGIC, 4);
  Serial.write((uint8_t*)&len, 4);     // ESP32 is little-endian
  Serial.write(fb->buf, fb->len);
  Serial.flush();

  esp_camera_fb_return(fb);
}

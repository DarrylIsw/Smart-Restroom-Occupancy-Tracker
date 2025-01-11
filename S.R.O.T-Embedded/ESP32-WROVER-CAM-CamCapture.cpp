// This code cannot run in .cpp, copy the code and paste it into an .ino file

#include "WiFi.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "HTTPClient.h"
#include "esp32-hal-cpu.h"
#include "esp_sleep.h"
#include <WebServer.h>
#include <WiFiClientSecure.h>
#include <Base64.h>
#include "time.h"
#include <queue>             // For the queue
#include <mutex>   

// Wi-Fi Credentials
const char* ssid = "RMX1971";
const char* password = "12345678";

// PIR Sensor Pins
#define ENTRY_PIR_PIN 12  // Entry PIR sensor

// Camera Pins for ESP32-CAM WROVER
#define PWDN_GPIO_NUM -1   // Not used
#define RESET_GPIO_NUM -1  // Not used
#define XCLK_GPIO_NUM 21
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 19
#define Y4_GPIO_NUM 18
#define Y3_GPIO_NUM 5
#define Y2_GPIO_NUM 4
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// Variables for PIR sensors
bool entryTriggered = false;
bool isProcessing = false;

// Timing Configuration
unsigned long inactivityDuration = 10 * 60 * 1000;  // 10 minutes in milliseconds
unsigned long lastActivityTime = 0;
unsigned long lastCaptureTime = 0;
unsigned long debounceDelay = 2000;  // 2 seconds debounce
unsigned long lastWiFiCheck = 0;
unsigned long wifiCheckInterval = 5000;  // Check Wi-Fi every 5 seconds

RTC_DATA_ATTR int sequenceNumber = 0;  // Persistent across deep sleep

WebServer server(80);  // Create a web server object

// Define the task queue
std::queue<int> captureQueue; // Queue for capture tasks
std::mutex queueMutex;        // Mutex for thread safety (optional)
#define MAX_QUEUE_SIZE 10 

/// Function to add a task to the queue
void addToQueue(int sensorId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (captureQueue.size() < MAX_QUEUE_SIZE) {  // Prevent queue from growing too large
        captureQueue.push(sensorId);
        Serial.printf("Added task from sensor %d to the queue. Queue size: %d\n", sensorId, captureQueue.size());
    } else {
        Serial.println("Queue full, discarding new task.");
    }
}

// Function to process tasks from the queue
void processQueue() {
    std::lock_guard<std::mutex> lock(queueMutex); // Protect access to the queue

    if (!captureQueue.empty() && !isProcessing) {  // Process if the queue is not empty and no capture is in progress
        int sensorId = captureQueue.front();       // Get the next task
        captureQueue.pop();                        // Remove the task from the queue

        Serial.printf("Processing task from sensor %d. Queue size: %d\n", sensorId, captureQueue.size());

        // Capture and send the photo
        captureAndSendPhoto();
    }
}

void setupWiFi() {
  Serial.println("Attempting to connect to Wi-Fi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long startAttemptTime = millis();

  // Wait for connection
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) { // Timeout after 10 seconds
    Serial.print(".");
    delay(500);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi connected successfully!");
    Serial.print("Assigned IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to Wi-Fi. Check credentials or signal strength.");
  }
}

void maintainWiFiConnection() {
  static unsigned long lastReconnectAttempt = 0;
  unsigned long currentMillis = millis();

  // If Wi-Fi is disconnected
  if (WiFi.status() != WL_CONNECTED) {
    if (currentMillis - lastReconnectAttempt >= wifiCheckInterval) {
      lastReconnectAttempt = currentMillis;
      Serial.println("Wi-Fi lost. Reconnecting...");
      WiFi.disconnect(); // Clear current connection
      WiFi.begin(ssid, password); // Reattempt connection

      // Optional: Check if reconnection succeeds
      unsigned long reconnectStart = millis();
      while (WiFi.status() != WL_CONNECTED && millis() - reconnectStart < 5000) { // 5 seconds timeout
        Serial.print(".");
        delay(500);
      }
      if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nReconnected to Wi-Fi!");
        Serial.print("Assigned IP: ");
        Serial.println(WiFi.localIP());
      } else {
        Serial.println("\nReconnection attempt failed.");
      }
    }
  } else {
    static bool wasConnected = false;
    if (!wasConnected) {
      Serial.println("Wi-Fi is already connected.");
      Serial.print("Assigned IP: ");
      Serial.println(WiFi.localIP());
      wasConnected = true;
    }
  }
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  int retryCount = 0;
  esp_err_t err;

  do {
    err = esp_camera_init(&config);
    if (err != ESP_OK) {
      Serial.printf("Camera init failed with error 0x%x. Retrying...\n", err);
      delay(2000);  // Wait before retrying
      retryCount++;
    }
  } while (err != ESP_OK && retryCount < 3);  // Retry 3 times

  if (err == ESP_OK) {
    Serial.println("Camera initialized successfully.");
  } else {
    Serial.println("Camera failed to initialize after 3 attempts.");
    // Optionally restart or enter safe mode
    ESP.restart();
  }
}

// Synchronize time with NTP servers
void syncTime() {
    configTime(0, 0, "pool.ntp.org", "time.nist.gov");
    Serial.println("Synchronizing time...");
    struct tm timeinfo;
    for (int retries = 0; retries < 10; retries++) {
        if (getLocalTime(&timeinfo)) {
            Serial.println("Time synchronized successfully!");
            return;
        }
        Serial.println("Failed to sync time, retrying...");
        delay(2000);
    }
    Serial.println("Failed to synchronize time after 10 attempts.");
}

void setup() {
  Serial.begin(115200);

  // Configure PIR pins
  pinMode(ENTRY_PIR_PIN, INPUT);

  // Enable wake-up on PIR sensor triggers
  uint64_t wakeupPins = (1ULL << ENTRY_PIR_PIN);
  esp_sleep_enable_ext1_wakeup(wakeupPins, ESP_EXT1_WAKEUP_ANY_HIGH);

  // Check wake-up reason
  if (esp_sleep_get_wakeup_cause() == ESP_SLEEP_WAKEUP_EXT1) {
    uint64_t wakeupPinMask = esp_sleep_get_ext1_wakeup_status();
    if (wakeupPinMask & (1ULL << ENTRY_PIR_PIN)) {
      Serial.println("Woke up due to ENTRY PIR sensor!");
    }
    lastActivityTime = millis();  // Reset activity timer
  } else {
    Serial.println("Normal boot.");
  }

  setupWiFi();  // Non-blocking Wi-Fi setup
  setupCamera();
  // setupWebServer();
  // syncTime();
  setCpuFrequencyMhz(240);      // Set CPU clock to 240MHz (maximum frequency for ESP32)
  lastActivityTime = millis();  // Start the inactivity timer
}

void clearCameraBuffer() {
    Serial.println("Clearing camera buffer...");
    int framesCleared = 0; // Counter for cleared frames
    const int maxFramesToClear = 3; // Maximum number of frames to clear

    while (framesCleared < maxFramesToClear) {
        camera_fb_t *fb = esp_camera_fb_get(); // Get a frame buffer
        if (!fb) {
            // If no more frames are available, break out of the loop
            Serial.println("No more frames in the buffer.");
            break;
        }
        esp_camera_fb_return(fb); // Release the buffer
        Serial.println("Released a frame buffer.");
        framesCleared++;
    }

    if (framesCleared >= maxFramesToClear) {
        Serial.println("Reached maximum frames to clear.");
    }
    Serial.println("Camera buffer cleared.");
}


void captureAndSendPhoto() {
    // Set processing flag to true
    isProcessing = true;

    // Clear stale buffers (if needed)
    clearCameraBuffer(); // Clear any stale buffers
    delay(100);          // Allow the camera time to stabilize

    // Capture a new frame
    camera_fb_t *fb = esp_camera_fb_get(); // Capture a new frame
    if (!fb) {
        Serial.println("Camera capture failed");
        isProcessing = false; // Reset flag if capture fails
        return;
    }

    if (fb->len == 0 || fb->buf == nullptr) {
        Serial.println("Captured image is empty or invalid. Aborting send.");
        esp_camera_fb_return(fb); // Free framebuffer before returning
        isProcessing = false;     // Reset flag
        return;
    }

    // Encode and send the image (Wi-Fi must be connected)
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        WiFiClientSecure client;
        client.setInsecure();  // Disable SSL verification

        String serverUrl = "http://192.168.27.1:5000/process-image";
        // String serverUrl = "https://qktmd56g-5000.asse.devtunnels.ms/process-image";

        // Base64 encode the image data
        String base64Image = base64::encode(fb->buf, fb->len); // Encode BEFORE freeing the buffer

        if (base64Image.length() == 0) {
            Serial.println("Base64 encoding failed.");
            esp_camera_fb_return(fb); // Free framebuffer
            isProcessing = false;    // Reset flag
            return;
        }

        esp_camera_fb_return(fb); // Free the frame buffer after encoding

        // Get the current time as a formatted string
        time_t now;
        struct tm timeinfo;
        char timestamp[20]; // Buffer to hold the timestamp
        time(&now);
        localtime_r(&now, &timeinfo);
        strftime(timestamp, sizeof(timestamp), "%y/%m/%d %H:%M:%S", &timeinfo);

        // Construct the JSON payload
        String jsonPayload = "{";
        jsonPayload += "\"timestamp\":\"" + String(timestamp) + "\","; // Include the timestamp
        jsonPayload += "\"id\":\"1\","; // Sample sensor ID, adjust as necessary
        jsonPayload += "\"toilet_id\":\"1\",";  // Sample toilet_id, adjust as necessary
        jsonPayload += "\"floor_id\":\"1\",";  // Sample floor_id, adjust as needed
        jsonPayload += "\"building_id\":\"1\",";  // Sample building_id, adjust as needed
        jsonPayload += "\"gender_id\":\"1\","; // Sample gender_id, modify if needed
        jsonPayload += "\"image\":\"" + base64Image + "\"}";


        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");

        int httpResponseCode = http.POST(jsonPayload);

        // Log the server response
        if (httpResponseCode > 0) {
            Serial.print("HTTP Response code: ");
            Serial.println(httpResponseCode);
            String response = http.getString();
            Serial.println("Response payload: " + response);
        } else {
            Serial.print("Error code: ");
            Serial.println(httpResponseCode);
        }

        http.end();
    } else {
        Serial.println("WiFi disconnected. Cannot send photo.");
    }

    // Reset processing flag
    isProcessing = false;
}


void loop() {
    server.handleClient();     // Handle incoming HTTP requests
    maintainWiFiConnection();  // Check Wi-Fi periodically

    unsigned long currentMillis = millis();
    int entryState = digitalRead(ENTRY_PIR_PIN);

    // Static variables for debouncing and capturing
    static unsigned long lastEntryTime = 0;
    static unsigned long lastCaptureTime = 0;
    static unsigned long lastActivityTime = millis();  // Track the last activity
    static unsigned long lastWiFiPrintTime = 0;

    // Action flags to prevent multiple captures
    static bool actionTaken = false;

    // Log PIR state
    Serial.print("PIR Entry Sensor State: ");
    Serial.println(entryState == HIGH ? "HIGH (Detected)" : "LOW (No Detection)");

    // Print Wi-Fi state every 5 seconds
    if (currentMillis - lastWiFiPrintTime >= 5000) {
        lastWiFiPrintTime = currentMillis;
        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("Wi-Fi Status: Connected");
            Serial.print("IP Address: ");
            Serial.println(WiFi.localIP());
        } else {
            Serial.println("Wi-Fi Status: Disconnected");
        }
    }

    if (entryState == HIGH) {
        // Entry sensor triggered
        // Debounce the trigger and ensure multiple triggers don't cause overlapping captures
        if (!actionTaken && (currentMillis - lastCaptureTime > debounceDelay)) {
            Serial.println("Entry sensor triggered: Adding to queue...");
            addToQueue(ENTRY_PIR_PIN);  // Add the task to the queue
            lastCaptureTime = currentMillis;
            lastActivityTime = currentMillis;
            actionTaken = true;  // Prevent multiple triggers until reset
        }
    }

    // Reset action flag when entry sensor is LOW
    if (entryState == LOW) {
        actionTaken = false;
    }

    // Process the queue only if no task is currently being processed
    if (!isProcessing) {
        processQueue();
    }

    // Enter deep sleep after inactivity
    if (currentMillis - lastActivityTime >= inactivityDuration) {
        Serial.println("No activity for 10 minutes. Checking wake-up status before sleeping.");

        if (esp_sleep_get_wakeup_cause() != ESP_SLEEP_WAKEUP_EXT1) {
            Serial.println("Entering deep sleep. Waiting for motion...");
            delay(1000);  // Allow Serial output
            esp_deep_sleep_start();  // Enter deep sleep
        } else {
            Serial.println("Recently woke up with no motion. Skipping deep sleep for now.");
        }
    }
}

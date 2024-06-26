#define TPLINK

// WIFI ROUTER TPLINK
#ifdef TPLINK
const char* ssid     = "TP-Link_DD58";
const char* password = "55118316";
const char* hostname = "acker-car";
#endif

// FIJAR IP, PUERTA Y SUBRED (No usado si useStaticIP = false, como actualmente está)
// Actualmente es la red wifi quien asigna estos valores por defecto
IPAddress ip(192, 168, 0, 137);
IPAddress gateway(192, 168, 0, 1);
IPAddress subnet(255, 255, 255, 0);

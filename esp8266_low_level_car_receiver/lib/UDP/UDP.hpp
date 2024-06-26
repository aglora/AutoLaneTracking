//#define CASA
//#define MOVIL
//#define WEMOS
#define TPLINK

// UDP variables
WiFiUDP UDP;

unsigned int localPort = 1234;
unsigned int remotePort = 47269;

// ASIGNACIÓN IP AL PC WIFI CASA
#ifdef CASA
IPAddress remoteIP(192, 168, 100, 34);
// WEMOS : 192.168.100.98
#endif 
// ASIGNACIÓN IP AL PC DEL MOVIL
#ifdef MOVIL
IPAddress remoteIP(192, 168, 43, 62);
// WEMOS : 192.168.43.51
#endif 
// ASIGNACIÓN AL PC WEMOS WIFI
#ifdef WEMOS
IPAddress remoteIP(192,168,4,2);
// WEMOS : 192.168.4.1
#endif 
// ASIGNACIÓN AL PC TPLINK ROUTER WIFI
#ifdef TPLINK
IPAddress remoteIP(192,168,0,4);
// WEMOS : 192.168.0.109
#endif 

char packetBuffer[UDP_TX_PACKET_MAX_SIZE]; //buffer to hold incoming packet,

void ProcessPacket(String response)
{
   Serial.println(response);
}

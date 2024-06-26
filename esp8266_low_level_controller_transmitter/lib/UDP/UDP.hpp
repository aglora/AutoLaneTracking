#define TPLINK

// UDP variables
WiFiUDP UDP;

unsigned int localPort = 1234;
//unsigned int remotePort = 1234; // acker-car-wemos
unsigned int remotePort = 1233; // rpi-juan

// ASIGNACIÓN AL PC TPLINK ROUTER WIFI
#ifdef TPLINK
//IPAddress remoteIP(192,168,0,109); //acker-car
IPAddress remoteIP(192,168,0,156); //rpi-juan
// WEMOS : 192.168.0.115 //acker-controller
#endif 

char packetBuffer[UDP_TX_PACKET_MAX_SIZE]; //buffer to hold incoming packet,

void ProcessPacket(String response)
{
   Serial.println(response);
}

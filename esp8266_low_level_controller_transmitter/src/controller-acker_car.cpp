#include <Wire.h>
#include "PCF8591.h"
// WIFI
#include <ESP8266WiFi.h>
#include <WiFiUDP.h>
#include "config.h" // Sustituir con datos de red
#include "UDP.hpp"
#include "ESP8266_Utils.hpp"
#include "ESP8266_Utils_UDP.hpp"
// Extras
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//! SELECT WIFI ACCES POINT
// #define WEMOS_WIFI
#define ROUTER_WIFI

//* DEFINICION PINES
#define JOY_RIGHT_BUTTON 12 // D6
#define LUX_SWITCH_PIN 13   // D7

PCF8591 pcf8591(0x48, 0);

IRAM_ATTR void ISR();

uint8_t A0_trac, A0_trac_ant, A1_dir, A1_dir_ant, A2_speed, A2_speed_ant;
int BuzzerState, BuzzerState_ant, LuzSwitch, LuzSwitch_ant;
char msg_string[50];

void setup()
{
  Wire.begin();
  Serial.begin(115200);
  pcf8591.begin();

  //* WIFI / UDP
  Serial.println("STARTING WIFI CONECTION");
#ifdef WEMOS_WIFI
  ConnectWiFi_AP();
#endif
#ifdef ROUTER_WIFI
  ConnectWiFi_STA();
#endif
  ConnectUDP();
  Serial.println("WIFI OK");
  delay(100);

  //* PINES
  pinMode(JOY_RIGHT_BUTTON, INPUT);
  pinMode(LUX_SWITCH_PIN, INPUT);

  //* INT RUT LUX
  ////attachInterrupt(digitalPinToInterrupt(LUX_SWITCH_PIN), ISR, RISING);
}

void loop()
{
  //! SUPERFICIES CONTROL
  A0_trac = pcf8591.adc_raw_read(0);  // JOYSTICK TRCCION EJE X HORIZONTAL
  A1_dir = pcf8591.adc_raw_read(1);   // JOYSTICK DIRECTION EJE Y VERTICAL
  A2_speed = pcf8591.adc_raw_read(2); // JOYSTICK SPEED POTENCIOMETRO
  A2_speed = map(A2_speed, 0, 255, 0, 100);
  BuzzerState_ant = BuzzerState;
  BuzzerState = digitalRead(JOY_RIGHT_BUTTON);
  LuzSwitch_ant = LuzSwitch;
  LuzSwitch = digitalRead(LUX_SWITCH_PIN);

  //! ENVIO DE COMANDOS

  //* TRACCION
  if (A0_trac != A0_trac_ant)
  {
    A0_trac_ant = A0_trac;

    if ((A0_trac < 200) && (A0_trac > 100))
    {
      SendUDP_Packet("S");
      delay(1);
    }
    else if (A0_trac >= 200)
    {
      SendUDP_Packet("F");
      delay(1);
    }
    else if (A0_trac <= 100)
    {
      SendUDP_Packet("B");
      delay(1);
    }
  }

  //* DIRECCION
  if (A1_dir != A1_dir_ant)
  {
    A1_dir_ant = A1_dir;

    if ((A1_dir < 200) && (A1_dir > 100))
    {
      SendUDP_Packet("G");
      delay(1);
    }
    else if (A1_dir >= 200)
    {
      SendUDP_Packet("R");
      delay(1);
    }
    else if (A1_dir <= 100)
    {
      SendUDP_Packet("L");
      delay(1);
    }
  }

  //* SPEAKER
  if (BuzzerState_ant == LOW && BuzzerState == HIGH)
  {
    delay(20);
    BuzzerState = digitalRead(JOY_RIGHT_BUTTON);
    if (BuzzerState)
      SendUDP_Packet("Y");
  }
  if (BuzzerState_ant == HIGH && BuzzerState == LOW)
  {
    delay(20);
    BuzzerState = digitalRead(JOY_RIGHT_BUTTON);
    if (!BuzzerState)
      SendUDP_Packet("X");
  }

  //* LUCES
  if (LuzSwitch_ant == LOW && LuzSwitch == HIGH)
  {
    delay(20);
    LuzSwitch = digitalRead(LUX_SWITCH_PIN);
    if (LuzSwitch)
      SendUDP_Packet("N");
  }

  //* VELOCIDAD
  if (A2_speed != A2_speed_ant)
  {
    A2_speed_ant = A2_speed;
    snprintf(msg_string, sizeof(msg_string), "%d", (int)A2_speed);
    SendUDP_Packet(msg_string);
    // SendUDP_Packet(String(A2_speed));
    delay(1);
  }
}

/* IRAM_ATTR void ISR()
{
    SendUDP_Packet("N");
    delay(20);
} */
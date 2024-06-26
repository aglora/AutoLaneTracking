/// CONTROL MANUAL ACKER-CAR

//* DEPENDENCIAS EXTERNAS
// FRAMEWORK ARDUINO
#include <Arduino.h>
// SPI
#include <Wire.h>
#include <SPI.h>
// WIFI
#include <ESP8266WiFi.h>
#include <WiFiUDP.h>
#include "config.h" // Sustituir con datos de red
#include "UDP.hpp"
#include "ESP8266_Utils.hpp"
#include "ESP8266_Utils_UDP.hpp"
// Servos Cola
#include <Servo.h>
// Calculos
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//! SELECT WIFI ACCES POINT
// #define WEMOS_WIFI
#define ROUTER_WIFI

//* DEFINICION PINES
#define IN2 5     // D1
#define IN1 4     // D2
#define IN4 0     // D3
#define IN3 2     // D4
#define ENA 14    // D5
#define ENB 12    // D6
#define BUZZER 13 // D7
#define LUZ 15    // D8

//* CABECERAS FUNCIONES Y SUBRUTINAS
void receiveCommand();
void speakerBasic();
void tocarNota(int nota, int duracion);

//* VARIABLES GLOBALES
int motor = 50; // Velocidad motor delantero %
int speed = 0;  // Velocidad 0-255 avance tracción
int dir = 0;    // Direccion 0-255 velocidad giro
bool forward = true;
bool left = false;
bool stop = true;
bool giro = false;
// variables UDP commands
// Secuencia de tonos agradable
// int notas[] = {262, 294, 330, 349, 392, 440, 494, 523}; // Notas correspondientes a las teclas blancas del piano (Do, Re, Mi, Fa, Sol, La, Si, Do)
int notas[] = {300, 1000}; // Notas correspondientes a las teclas blancas del piano (Do, Re, Mi, Fa, Sol, La, Si, Do)
enum Estado
{
  ESPERANDO,
  REPRODUCIENDO_NOTAS
};
Estado estadoActual = ESPERANDO;
unsigned long tiempoInicio = 0;
unsigned long tiempoTranscurrido = 0;
int indiceNota = 0;
int duracionNota = 100;
// luces
bool leds = false;
//int LDR_value;

//* CONFIGURACION INICIAL DEL SISTEMA
void setup()
{
  //* PUERTO SERIE
  Serial.begin(115200);
  while (!Serial)
    delay(10);

  //* PINES
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(LUZ, OUTPUT);

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
}

//* FUNCION PRINCIPAL
void loop()
{
  //! RECIBIR COMANDOS
  receiveCommand();

  //! DIRECCION (MOTOR DELANTERO)
  if (giro) // SE DARA GIRO
  {
    // SENTIDO
    if (left) // IZQUIERDA
    {
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
    }
    else // DERECHA
    {
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
    }
    dir = 255;
    analogWrite(ENA, dir);
  }
  else // NO SE DARA GIRO
  {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    digitalWrite(ENA, 0);
  }

  //! TRACCION (MOTOR TRASERO)
  if (!stop) // ACTIVADO
  {
    // SENTIDO
    if (forward) // AVANCE
    {
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    }
    else // RETROCESO
    {
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
    }
    speed = map(motor, 0, 100, 40, 255);
    analogWrite(ENB, speed);
  }
  else // PARADO
  {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
    digitalWrite(ENB, 0);
  }

  //! SPEAKER
  /*   if (motor >= 80)
      estadoActual = REPRODUCIENDO_NOTAS;
    else
      estadoActual = ESPERANDO; */
  speakerBasic();

  //! LDR
  //LDR_value = analogRead(A0);
  bool LDR_value = digitalRead(16);
/*   Serial.print("LDR: ");
  Serial.println(LDR_value); */

  //! LUCES
  //if (leds == true || LDR_value <= 200)
  if (leds == true || LDR_value == false)
  //if (leds == true)
    digitalWrite(LUZ, HIGH);
  else
    digitalWrite(LUZ, LOW);

  //! COMPROBACIÓN CONEXION WIFI
  if (WiFi.status() != WL_CONNECTED)
  {
    stop = true;
    giro = false;
  }
}

//* FUNCIONES AUXILIARES Y SUBRUTINAS
void receiveCommand()
{
  int packetSize = UDP.parsePacket();
  if (packetSize)
  {
    // read the packet into packetBufffer
    memset(packetBuffer, 0, UDP_TX_PACKET_MAX_SIZE);
    UDP.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);

    Serial.println();
    Serial.print("Received packet of size ");
    Serial.print(packetSize);
    Serial.print(" from ");
    Serial.print(UDP.remoteIP());
    Serial.print(":");
    Serial.println(UDP.remotePort());
    Serial.print("Payload: ");
    Serial.write((uint8_t *)packetBuffer, (size_t)packetSize);
    Serial.println();
    delay(10);

    //* TRACCION
    if (packetBuffer[0] == 'S')
    {
      stop = true;
    }
    else if (packetBuffer[0] == 'F')
    {
      stop = false;
      forward = true;
    }
    else if (packetBuffer[0] == 'B')
    {
      stop = false;
      forward = false;
    }
    //* DIRECCION
    else if (packetBuffer[0] == 'G')
    {
      giro = false;
    }
    else if (packetBuffer[0] == 'R')
    {
      giro = true;
      left = false;
    }
    else if (packetBuffer[0] == 'L')
    {
      giro = true;
      left = true;
    }
    //* SPEAKER
    else if (packetBuffer[0] == 'X')
    {
      estadoActual = REPRODUCIENDO_NOTAS;
    }
    else if (packetBuffer[0] == 'Y')
    {
      estadoActual = ESPERANDO;
    }
    //* LUCES
    else if (packetBuffer[0] == 'N')
    {
      if (leds)
        leds = false;
      else
        leds = true;
    }
    //* VELOCIDAD
    else
    {
      motor = atoi(packetBuffer);
      Serial.println(packetBuffer);
    }
  }
}

void speakerBasic()
{
  switch (estadoActual)
  {
  case ESPERANDO:
    break;

  case REPRODUCIENDO_NOTAS:
    tiempoTranscurrido = millis() - tiempoInicio;

    if (tiempoTranscurrido >= duracionNota)
    {
      tocarNota(notas[indiceNota], 0); // Detener la nota actual
      indiceNota++;

      if (indiceNota >= 3)
      {
        // Se completó la secuencia de notas
        indiceNota = 0;
      }
      else
      {
        tocarNota(notas[indiceNota], duracionNota);
        tiempoInicio = millis();
      }
    }
    break;
  }
}

// Función para reproducir una nota en el buzzer
void tocarNota(int nota, int duracion)
{
  if (nota == 0)
  {
    noTone(BUZZER);
  }
  else
  {
    tone(BUZZER, nota, duracion);
  }
  delay(duracion);
  noTone(BUZZER);
}
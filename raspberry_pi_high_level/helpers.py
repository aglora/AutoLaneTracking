import socket
import userConfig
import time

comando_coche = "G"
active_turn = False


# Funcion para agregar un comando a la lista de manera segura
def agregar_comando(lista_comandos, comando, lock):
    with lock:
        lista_comandos.append(comando)
    return lista_comandos


def recibir_comandos():
    global comando_coche
    lista_comandos = []
    modoControl = True  # Comenzar en modo manual

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((userConfig.UDP_IP_mando, userConfig.UDP_PORT_rpi))

    while True:
        # Recepción y lectura de comandos del mando
        data, addr = sock.recvfrom(1024)
        comando_mando = data.decode("utf-8")

        if comando_mando == "X":  # R3 -> Cambio de estado entre modo manual/autónomo
            lista_comandos.append("S")  # Orden comando de parada
            modoControl = not modoControl
            if modoControl:
                print("CAMBIO MODO CONTROL: Manual")
            else:
                print("CAMBIO MODO CONTROL: Autonomo")
        if modoControl:  # True -> Manual
            lista_comandos.append(comando_mando)
        else:  # modoControl: False -> Autónomo
            lista_comandos.append(comando_coche)
            if active_turn:
                lista_comandos.append(str(userConfig.MIN_SPEED_TURN))
            else:
                lista_comandos.append(str(userConfig.MIN_SPEED_FORWARD))
            lista_comandos.append("F")

        if lista_comandos != []:  # Enviar comandos hasta vaciar buffer
            for comando in lista_comandos:
                sock.sendto(
                    bytes(comando, "utf-8"),
                    (userConfig.UDP_IP_wemos, userConfig.UDP_PORT_wemos),
                )
            lista_comandos = []

        time.sleep(0.01)  # frecuencia de 100 Hz

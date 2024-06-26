import userConfig
import time
import cv2
import numpy as np
import skimage.morphology as morph
import socket
import threading
import os
import helpers
# import keyboard

# ----------------------------------------------------------------
# stop_program = False


# def key_stop_check(key):
#    global stop_program
#    if key.name == "q":
#        stop_program = True
#        keyboard.unhook_all()


hilo_comandos = threading.Thread(target=helpers.recibir_comandos, daemon=True)
hilo_comandos.start()
# keyboard.on_press(key_stop_check)

# Abrir el archivo de video de entrada
if userConfig.SIMULATION_MODE == True:
    input_video = cv2.VideoCapture("vista_camara_baja_coche.mp4")
    # Comprobar si el video se ha abierto correctamente
    if not input_video.isOpened():
        print("Error: No se pudo abrir el archivo de video.")
        exit()
    # crear o limpiar contenido del registro de curvatura
    with open("curvatura_register.txt", "w"):
        pass

    # Obtener las dimensiones del video de entrada
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Frame Height: ", frame_height)
    print("Frame Width: ", frame_width)
else:
    # Use real camera of the car
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la camara")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, userConfig.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, userConfig.CAMERA_HEIGHT)
    frame_width = userConfig.CAMERA_WIDTH
    frame_height = userConfig.CAMERA_HEIGHT
    print("Frame Height: ", frame_height)
    print("Frame Width: ", frame_width)
frame_interval = 1 / userConfig.LOOP_FREQ

# # Definir el codec y crear el objeto VideoWriter
# output_video = cv2.VideoWriter(
#     "output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height)
# )

while (userConfig.SIMULATION_MODE and input_video.isOpened()) or (
    not userConfig.SIMULATION_MODE
):
    if userConfig.SIMULATION_MODE:
        ret, frame = input_video.read()
        if not ret:
            break
    else:
        ret, frame = cap.read()
        if not ret:
            break

    # if userConfig.DEBUG_MODE:
    #     cv2.imshow("originalFrame", frame)
    #     cv2.waitKey(1)
    # Binarizar la imagen

    # Binarizar la imagen
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if userConfig.DEBUG_MODE:
    #     cv2.imshow("greyFrame", grey_frame)
    #     cv2.waitKey(1)
    ret, thresholded_frame = cv2.threshold(
        grey_frame, userConfig.MIN_V, userConfig.MAX_V, cv2.THRESH_BINARY
    )
    # if userConfig.DEBUG_MODE:
    #     cv2.imshow("thresholdedFrame", thresholded_frame)
    #     cv2.waitKey(1)

    if userConfig.MORPHOLOGICAL_OPERATIONS_ON == True:
        # Aplicar el filtro morfológico
        kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
        opened_frame = cv2.morphologyEx(
            thresholded_frame, cv2.MORPH_OPEN, kernel, iterations=1
        )
        cv2.bitwise_not(opened_frame, opened_frame)
        # if userConfig.DEBUG_MODE:
        #     cv2.imshow("openedFrame", opened_frame)
        #     cv2.waitKey()

        # Operación de dilatación
        dilation_kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
        dilated_frame = cv2.dilate(opened_frame, dilation_kernel, iterations=1)
        # if userConfig.DEBUG_MODE:
        #     cv2.imshow("dilatedFrame", dilated_frame)
        #     cv2.waitKey()

        # Operación de erosión
        erosion_kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
        eroded_frame = cv2.erode(dilated_frame, erosion_kernel, iterations=1)
        # if userConfig.DEBUG_MODE:
        cv2.imshow("processedImage", eroded_frame)
        cv2.waitKey(1)
    else:
        eroded_frame = thresholded_frame

    # Encontrar las regiones etiquetadas y seleccionar la más larga en el eje x
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded_frame
    )
    # if userConfig.DEBUG_MODE:
    #     eroded_frameColor = cv2.cvtColor(eroded_frame, cv2.COLOR_GRAY2RGB)
    #     for centroid in centroids:
    #         centroid = [int(element) for element in centroid]
    #         frame_infoLabelled = cv2.circle(
    #             eroded_frameColor,
    #             tuple(centroid),
    #             radius=10,
    #             color=(255, 0, 0),
    #             thickness=5,
    #         )
    #         # cv2.imshow("erodedFrame", frame_infoLabelled)
    #         # cv2.waitKey()

    # Calcular el límite de la mitad inferior de la imagen
    y_threshold = userConfig.Y_THRESHOLD
    # Dibujar una línea horizontal
    cv2.line(
        frame, (0, y_threshold), (frame.shape[1], y_threshold), (0, 255, 0), thickness=1
    )

    # Crear una lista de índices de regiones ordenadas por extensión en el eje x (de mayor a menor)
    sorted_regions = sorted(
        range(1, num_labels), key=lambda x: stats[x, cv2.CC_STAT_WIDTH], reverse=True
    )

    # Buscar la región que cumple con los criterios
    valid_regions = []
    max_skeleton = 0
    for label in sorted_regions:
        # Obtener el área y la relación de aspecto de la región
        # x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        # aspect_ratio = float(w) / h

        area = stats[label, cv2.CC_STAT_AREA]
        skeleton_line = labels == label
        skeleton_pixels = np.sum(skeleton_line)
        pixels_below_threshold = np.sum(skeleton_line[y_threshold:, :])
        if (
            pixels_below_threshold
            >= userConfig.MIN_LINE_LENGTH_PERCENTAGE * skeleton_pixels
            and skeleton_pixels > 30
            and area > userConfig.MIN_REGION_AREA
            # and aspect_ratio > userConfig.ASPECT_RATIO
        ):
            valid_regions.append(label)

            # Dibujar la región
            region_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 15, 200), thickness=cv2.FILLED)
            if skeleton_pixels > max_skeleton:
                max_skeleton = skeleton_pixels
                largest_label = label

    masked_frame = frame
    if valid_regions:

        # Crear una máscara para la región más larga
        largest_region_mask = np.uint8(labels == largest_label)

        # Operación de Zhang-Suen para esqueletizar las regiones
        skeletonized = (
            morph.skeletonize(largest_region_mask, method="zhang").astype(np.uint8)
            * 255
        )

        # Ajustar polinomio a curva
        lines_coordinates = np.argwhere(skeletonized != 0)
        if lines_coordinates.any():
            x_array = [point[1] for point in lines_coordinates]
            y_array = [frame_height - point[0] for point in lines_coordinates]
            if userConfig.DEBUG_MODE:
                matrix = np.zeros((frame_height, frame_width, 3), dtype=int)
                for x, y in zip(x_array, y_array):
                    matrix[int(frame_height - 1 - y), x, 0] = 255
                    matrix[int(frame_height - 1 - y), x, 1] = 0
                    matrix[int(frame_height - 1 - y), x, 2] = 0
                matrix = matrix.astype(frame.dtype)
                data_frame = cv2.addWeighted(frame, 1, matrix, 1, 0)
                # cv2.imshow("frame", data_frame)
                # cv2.waitKey()

        coefficients_line = np.polyfit(x_array, y_array, 1)
        coefficients_curve = np.polyfit(x_array, y_array, userConfig.POLY_DEGREE)
        line_polynomial = np.poly1d(coefficients_line)
        curve_polynomial = np.poly1d(coefficients_curve)

        # Calcular la pendiente
        first_derivative = np.polyder(line_polynomial, 1)
        pendiente = np.polyval(first_derivative, 0)
        print("count: ", len(valid_regions))
        if len(valid_regions) == 1:  # TODO
            helpers.active_turn = True
            if (
                pendiente > 0
            ):  # TODO: PLANTEAR MEJOR YA QUE DEPENDE DE LA DETECCIÓN LINEA DERECHA O IZQ
                helpers.comando_coche = "R"
                # print("GIRE PER LA DETRA")
                # giro a derecha
            else:
                # giro a izq
                # print("GIRE PER LA QUERDA")
                helpers.comando_coche = "L"
        elif len(valid_regions) != 0:
            helpers.active_turn = False
            helpers.comando_coche = "G"

        # Calcular la curvatura media
        # x_values = np.linspace(0, frame_width - 1, 10)
        # second_derivative = np.polyder(curve_polynomial, 2)
        # curvatures = np.polyval(second_derivative, x_values)
        # mean_curvature = np.mean(np.abs(curvatures))

        if userConfig.DEBUG_MODE == True:
            text = "Regiones: " + str("{:.2f}".format(len(valid_regions)))
            position = (70, 220)  # (x, y) coordinates of the top-left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 250, 0)  # White color
            thickness = 2

        # Draw the text on the image
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            font_color,
            thickness,
        )

        text = "Pendiente: " + str("{:.2f}".format(pendiente))
        position = (70, 200)  # (x, y) coordinates of the top-left corner
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            font_color,
            thickness,
        )

        # if userConfig.DEBUG_MODE == True:
        #     print("Curvatura: ", mean_curvature)
        #     with open("curvatura_register.txt", "a") as file:
        #         # Añadir el valor del array en una nueva línea
        #         # Convertir el array a una cadena de texto y añadir una nueva línea
        #         file.write(str(mean_curvature) + ", " + str(pendiente) + "\n")

        if userConfig.DEBUG_MODE == True:
            # Crear una imagen verde del mismo tamaño que skeletonized
            green_skeletonized = np.zeros_like(frame)
            # Establecer los píxeles esqueletizados en verde
            green_skeletonized[skeletonized != 0] = [0, 255, 0]

        # Aplicar la máscara sobre el fotograma original
        masked_frame = cv2.addWeighted(frame, 1, green_skeletonized, 1, 0)
        if userConfig.DEBUG_MODE:
            # Dibujar curva polinómica
            x_values = np.arange(0, frame_width - 1)
            y_values_line = line_polynomial(x_values)
            y_values_line = np.round(y_values_line).astype(int)
            y_values_line = np.clip(y_values_line, 0, frame_height - 1)
            y_values_curve = curve_polynomial(x_values)
            y_values_curve = np.round(y_values_curve).astype(int)
            y_values_curve = np.clip(y_values_curve, 0, frame_height - 1)

        # Representar línea estimada
        matrix_line = np.zeros((frame_height, frame_width, 3), dtype=int)
        for x, y in zip(x_values, y_values_line):
            matrix_line[int(frame_height - 1 - y), x, 0] = 255
            matrix_line[int(frame_height - 1 - y), x, 1] = 0
            matrix_line[int(frame_height - 1 - y), x, 2] = 0

            matrix_line = matrix_line.astype(masked_frame.dtype)
            masked_frame = cv2.addWeighted(masked_frame, 1, matrix_line, 1, 0)

        # Representar curva estimada
        matrix_curve = np.zeros((frame_height, frame_width, 3), dtype=int)
        for x, y in zip(x_values, y_values_curve):
            matrix_curve[int(frame_height - 1 - y), x, 0] = 0
            matrix_curve[int(frame_height - 1 - y), x, 1] = 0
            matrix_curve[int(frame_height - 1 - y), x, 2] = 255

        matrix_curve = matrix_curve.astype(masked_frame.dtype)
        masked_frame = cv2.addWeighted(masked_frame, 1, matrix_curve, 1, 0)

    if userConfig.DEBUG_MODE:
        cv2.imshow("maskedFrame", masked_frame)
        cv2.waitKey(1)

    time.sleep(frame_interval)

cv2.destroyAllWindows()
if userConfig.SIMULATION_MODE:
    # Liberar los recursos
    input_video.release()
else:
    cap.release()


frame_count = 0


# Incrementar el contador de fotogramas procesados
frame_count += 1

#     # Binarizar la imagen
#     if userConfig.DEBUG_MODE:
#         cv2.imshow("originalFrame", frame)
#         cv2.waitKey()
#     # Binarizar la imagen
#     grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     ret, thresholded_frame = cv2.threshold(
#         grey_frame, userConfig.MIN_V, userConfig.MAX_V, cv2.THRESH_BINARY
#     )
#     if userConfig.DEBUG_MODE:
#         cv2.imshow("thresholdedFrame", thresholded_frame)
#         cv2.waitKey()

#     if userConfig.MORPHOLOGICAL_OPERATIONS_ON == True:

#         # Aplicar el filtro morfológico
#         kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
#         opened_frame = cv2.morphologyEx(
#             thresholded_frame, cv2.MORPH_OPEN, kernel, iterations=1
#         )
#         cv2.bitwise_not(opened_frame, opened_frame)
#         if userConfig.DEBUG_MODE:
#             cv2.imshow("openedFrame", opened_frame)
#             cv2.waitKey()

#         # Operación de dilatación
#         dilation_kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
#         dilated_frame = cv2.dilate(opened_frame, dilation_kernel, iterations=1)
#         if userConfig.DEBUG_MODE:
#             cv2.imshow("dilatedFrame", dilated_frame)
#             cv2.waitKey()

#         # Operación de erosión
#         erosion_kernel = np.ones(userConfig.KERNEL_SHAPE, np.uint8)
#         eroded_frame = cv2.erode(dilated_frame, erosion_kernel, iterations=1)
#         if userConfig.DEBUG_MODE:
#             cv2.imshow("erodedFrame", eroded_frame)
#             cv2.waitKey()
#     else:
#         eroded_frame = thresholded_frame

#     frame = frame

#     # Encontrar las regiones etiquetadas y seleccionar la más larga en el eje x
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         eroded_frame
#     )
#     if userConfig.DEBUG_MODE:
#         eroded_frameColor = cv2.cvtColor(eroded_frame, cv2.COLOR_GRAY2RGB)
#         for centroid in centroids:
#             centroid = [int(element) for element in centroid]
#             frame_infoLabelled = cv2.circle(
#                 eroded_frameColor,
#                 tuple(centroid),
#                 radius=10,
#                 color=(255, 0, 0),
#                 thickness=5,
#             )
#             cv2.imshow("erodedFrame", frame_infoLabelled)
#             cv2.waitKey()

#     # Calcular el límite de la mitad inferior de la imagen
#     y_threshold = frame.shape[0] // 3

#     # Crear una lista de índices de regiones ordenadas por extensión en el eje x (de mayor a menor)
#     sorted_regions = sorted(
#         range(1, num_labels), key=lambda x: stats[x, cv2.CC_STAT_WIDTH], reverse=True
#     )

#     # Buscar la región que cumple con los criterios
#     for label in sorted_regions:
#         skeleton_line = labels == label
#         skeleton_pixels = np.sum(skeleton_line)
#         pixels_below_threshold = np.sum(skeleton_line[y_threshold:, :])
#         if pixels_below_threshold >= 0.2 * skeleton_pixels:
#             largest_label = label
#             break
#     else:
#         largest_label = None

#     # Crear una máscara para la región más larga
#     largest_region_mask = np.uint8(labels == largest_label)

#     # Operación de Zhang-Suen para esqueletizar las regiones
#     skeletonized = (
#         morph.skeletonize(largest_region_mask, method="zhang").astype(np.uint8) * 255
#     )

#     # Ajustar polinomio a curva
#     masked_frame = frame
#     lines_coordinates = np.argwhere(skeletonized != 0)
#     if lines_coordinates.any():
#         x_array = [point[1] for point in lines_coordinates]
#         y_array = [frame_height - point[0] for point in lines_coordinates]
#         if userConfig.DEBUG_MODE:
#             matrix = np.zeros((frame_height, frame_width, 3), dtype=int)
#             for x, y in zip(x_array, y_array):
#                 matrix[int(frame_height - 1 - y), x, 0] = 255
#                 matrix[int(frame_height - 1 - y), x, 1] = 0
#                 matrix[int(frame_height - 1 - y), x, 2] = 0
#             matrix = matrix.astype(frame.dtype)
#             data_frame = cv2.addWeighted(frame, 1, matrix, 1, 0)
#             cv2.imshow("frame", data_frame)
#             cv2.waitKey()

#         coefficients_line = np.polyfit(x_array, y_array, 1)
#         coefficients_curve = np.polyfit(x_array, y_array, userConfig.POLY_DEGREE)
#         line_polynomial = np.poly1d(coefficients_line)
#         curve_polynomial = np.poly1d(coefficients_curve)

#         # Calcular la pendiente
#         first_derivative = np.polyder(line_polynomial, 1)
#         pendiente = np.polyval(first_derivative, 0)
#         if (abs(pendiente) < 0.2) or ((abs(pendiente) < 0.25) and active_turn):  # TODO
#             active_turn = True
#             if (
#                 pendiente > 0
#             ):  # TODO: PLANTEAR MEJOR YA QUE DEPENDE DE LA DETECCIÓN LINEA DERECHA O IZQ
#                 helpers.comando_coche = "R"
#                 # giro a derecha
#             else:
#                 # giro a izq
#                 helpers.comando_coche = "L"

#                 pass

#             # comandar giro
#         else:
#             active_turn = False

#         # Calcular la curvatura media
#         x_values = np.linspace(0, frame_width - 1, 10)
#         second_derivative = np.polyder(curve_polynomial, 2)
#         curvatures = np.polyval(second_derivative, x_values)
#         mean_curvature = np.mean(np.abs(curvatures))
#         if userConfig.SIMULATION_MODE == True:
#             text = "Curvatura: " + str(mean_curvature)
#             position = (10, 30)  # (x, y) coordinates of the top-left corner
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 1
#             font_color = (255, 255, 255)  # White color
#             thickness = 2

#             # Draw the text on the image
#             cv2.putText(
#                 frame,
#                 text,
#                 position,
#                 font,
#                 font_scale,
#                 font_color,
#                 thickness,
#             )

#             text = "Pendiente: " + str(pendiente)
#             position = (10, 80)  # (x, y) coordinates of the top-left corner
#             cv2.putText(
#                 frame,
#                 text,
#                 position,
#                 font,
#                 font_scale,
#                 font_color,
#                 thickness,
#             )

#             if userConfig.DEBUG_MODE == True:
#                 print("Curvatura: ", mean_curvature)
#                 with open("curvatura_register.txt", "a") as file:
#                     # Añadir el valor del array en una nueva línea
#                     # Convertir el array a una cadena de texto y añadir una nueva línea
#                     file.write(str(mean_curvature) + ", " + str(pendiente) + "\n")

#             if userConfig.SIMULATION_MODE == True:
#                 # Crear una imagen verde del mismo tamaño que skeletonized
#                 green_skeletonized = np.zeros_like(frame)
#                 # Establecer los píxeles esqueletizados en verde
#                 green_skeletonized[skeletonized != 0] = [0, 255, 0]

#                 # Aplicar la máscara sobre el fotograma original
#                 masked_frame = cv2.addWeighted(
#                     frame, 1, green_skeletonized, 1, 0
#                 )
#                 if userConfig.SIMULATION_MODE:
#                     # Dibujar curva polinómica
#                     x_values = np.arange(0, frame_width - 1)
#                     y_values_line = line_polynomial(x_values)
#                     y_values_line = np.round(y_values_line).astype(int)
#                     y_values_line = np.clip(y_values_line, 0, frame_height - 1)
#                     y_values_curve = curve_polynomial(x_values)
#                     y_values_curve = np.round(y_values_curve).astype(int)
#                     y_values_curve = np.clip(y_values_curve, 0, frame_height - 1)

#                     # Representar línea estimada
#                     matrix_line = np.zeros((frame_height, frame_width, 3), dtype=int)
#                     for x, y in zip(x_values, y_values_line):
#                         matrix_line[int(frame_height - 1 - y), x, 0] = 255
#                         matrix_line[int(frame_height - 1 - y), x, 1] = 0
#                         matrix_line[int(frame_height - 1 - y), x, 2] = 0

#                     matrix_line = matrix_line.astype(masked_frame.dtype)
#                     masked_frame = cv2.addWeighted(masked_frame, 1, matrix_line, 1, 0)

#                     # Representar curva estimada
#                     matrix_curve = np.zeros((frame_height, frame_width, 3), dtype=int)
#                     for x, y in zip(x_values, y_values_curve):
#                         matrix_curve[int(frame_height - 1 - y), x, 0] = 0
#                         matrix_curve[int(frame_height - 1 - y), x, 1] = 0
#                         matrix_curve[int(frame_height - 1 - y), x, 2] = 255

#                     matrix_curve = matrix_curve.astype(masked_frame.dtype)
#                     masked_frame = cv2.addWeighted(masked_frame, 1, matrix_curve, 1, 0)

#                     if userConfig.DEBUG_MODE:
#                         cv2.imshow("maskedFrame", masked_frame)
#                         cv2.waitKey()

#     # Escribir el fotograma procesado en el video de salida
#     output_video.write(masked_frame)

#     # Mostrar el porcentaje de procesamiento
#     print(
#         f"Procesando fotograma {frame_count} de {total_frames} ({frame_count / total_frames * 100:.2f}%)",
#         end="\r",
#     )
#     cv2.destroyAllWindows()
# end_time = time.time()
# print("Tiempo total de procesado: ", end_time - start_time)
# print("Frames totales procesados: ", total_frames)
# print("FPS medios: ", total_frames / (end_time - start_time))

# # Liberar los recursos
# input_video.release()
# output_video.release()

while True:
    pass

from ultralytics import YOLO
import sys
import os
# Cargar modelo — no modificar
modelo = YOLO("yolov8n.pt")

# Analizar imagen — no modificar
imagen = sys.argv[1]
resultados = modelo(imagen)

lineas = []

# 1. Extraer el primer (y único) resultado de la imagen procesada
resultado = resultados[0]

clases_detectadas = [resultado.names[int(cls_id)] for cls_id in resultado.boxes.cls]

# 3. Eliminar duplicados usando set() para cualquier etiqueta detectada
etiquetas_finales = list(set(clases_detectadas))

# 4. Formatear las líneas para el archivo
nombre_imagen = os.path.basename(imagen) # Extrae solo el nombre, ignorando la ruta si la hay
lineas.append(f"Imagen elegida: {nombre_imagen}\n")
lineas.append(f"Etiquetas detectadas: {etiquetas_finales}\n")

# Guardar — no modificar
with open("resultados.txt", "w", encoding="utf-8") as f:
    f.writelines(lineas)

print("✅ Resultados guardados en resultados.txt")
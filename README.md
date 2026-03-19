# Taller: Detección de Objetos con YOLOv8
## Descripción del escenario
Un vehículo autónomo cuenta con una cámara frontal que captura imágenes 
del entorno mientras se encuentra en movimiento. Estas imágenes 
permiten identificar elementos de la vía como vehículos, peatones y señales 
de tránsito.

## Requerimiento
El sistema debe detectar los objetos frente al vehículo para evitar colisiones 
y tomar decisiones seguras. Como estas decisiones deben tomarse en tiempo real, 
el procesamiento se realiza localmente en el vehículo aplicando Edge Computing.

## Pasos

### 1. Instalar la librería
pip install ultralytics

### 2. Estructura de carpetas
taller/
├── detector.py </br>
├── image1.jpeg / image2.jpeg / ... / image7.jpeg

### 3. Elegir una imagen para procesar
Cada pareja debe elegir UNA imagen de las disponibles.

### 4. Correr el código
python detector.py imageN.jpeg/png  ← reemplaza N por el número de tu imagen

### 5. Resultado esperado en resultados.txt
Imagen elegida: image1.jpeg
Etiquetas detectadas: ['car', 'person', 'traffic light']

## Integrantes
- Nombre 1
- Nombre 2

## Lo que entregan en el fork
taller/</br>
├── detector.py         ← código completo</br>
├── image.jpeg          ← la imagen elegida</br>
└── resultados.txt      ← generado al correr el código, incluye imagen elegida
```

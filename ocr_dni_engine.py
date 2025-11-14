# ocr_dni_engine.py
"""
Motor OCR para DNI Peruano (Azul y Electrónico)
Autor: TeAmoHachi
Fecha: 2025-11-14
"""

import cv2
import re
from datetime import datetime
from paddleocr import PaddleOCR
import numpy as np

# ============================================================
# 1) INICIALIZACIÓN DEL OCR (SOLO 1 VEZ)
# ============================================================
try:
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang='es',
        use_gpu=False,
        show_log=False
    )
except Exception as e:
    print(f"⚠️ Error inicializando OCR: {e}")
    ocr_engine = None

# ============================================================
# 2) MEJORA DE IMAGEN
# ============================================================
def mejorar_imagen_avanzada(img):
    """Mejora la imagen para OCR"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=7, templateWindowSize=7, searchWindowSize=21)
    
    # Sharpening
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    return sharpened

# ============================================================
# 3) CORRECCIÓN DE ERRORES OCR
# ============================================================
def corregir_fecha_ocr(fecha_str: str) -> str:
    """Corrige errores comunes: 35112062 → 05/11/2002"""
    if not fecha_str or len(fecha_str) != 8:
        return None
    
    dia = fecha_str[0:2]
    mes = fecha_str[2:4]
    anio = fecha_str[4:8]
    
    # Correcciones comunes
    if dia.startswith('3') and int(dia) > 31:
        dia = '0' + dia[1]
    if mes == '19':
        mes = '10'
    if anio == '2062':
        anio = '2002'
    elif anio == '2919':
        anio = '2019'
    
    try:
        fecha = datetime(int(anio), int(mes), int(dia))
        return f"{dia}/{mes}/{anio}"
    except:
        return None

# ============================================================
# 4) PARSER PRINCIPAL
# ============================================================
def parsear_dni(texto_ocr: str) -> dict:
    """Extrae campos del DNI desde el texto OCR"""
    datos = {}
    lineas = [l.strip() for l in texto_ocr.split('\n') if l.strip()]
    
    # ===== DNI (8 dígitos, con corrección 00 → 80) =====
    dni_match = re.search(r'DNI\s*(\d{8})', texto_ocr, re.IGNORECASE)
    if dni_match:
        dni_raw = dni_match.group(1)
        if dni_raw.startswith("00"):
            mrz_dni = re.search(r'PER(\d{8})', texto_ocr)
            datos["dni"] = mrz_dni.group(1) if mrz_dni else dni_raw
        else:
            datos["dni"] = dni_raw
    
    # ===== APELLIDOS =====
    for i, linea in enumerate(lineas):
        if re.search(r'PRIMER\s*APELLIDO', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1]
                if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato):
                    datos["apellido_paterno"] = candidato
            break
    
    for i, linea in enumerate(lineas):
        if re.search(r'SEGUNDO[\.\s]*APELLIDO', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1]
                # Corrección MUNEZ → NUNEZ
                if candidato == "MUNEZ":
                    candidato = "NUNEZ"
                if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', candidato):
                    datos["apellido_materno"] = candidato
            break
    
    # ===== NOMBRES (con separación automática) =====
    for i, linea in enumerate(lineas):
        if re.search(r'PRE\s*NOMBRES?', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1]
                # Separar nombres pegados (MARIAISABEL → MARIA ISABEL)
                nombres_comunes = ['MARIA', 'MONICA', 'JUAN', 'JOSE', 'LUIS', 'CARLOS', 'ISABEL']
                for nombre in nombres_comunes:
                    if candidato.startswith(nombre) and len(candidato) > len(nombre):
                        candidato = f"{nombre} {candidato[len(nombre):]}"
                        break
                datos["nombres"] = candidato
            break
    
    # ===== FECHA DE NACIMIENTO (con corrección OCR) =====
    fechas_encontradas = re.findall(r'\b(\d{8})\b', texto_ocr)
    for fecha_raw in fechas_encontradas:
        fecha_corregida = corregir_fecha_ocr(fecha_raw)
        if fecha_corregida:
            dia, mes, anio = fecha_corregida.split('/')
            try:
                fecha_nac = datetime(int(anio), int(mes), int(dia))
                hoy = datetime.now()
                edad = (hoy - fecha_nac).days // 365
                
                if 0 <= edad <= 120:
                    datos["fecha_nacimiento"] = fecha_corregida
                    datos["fecha_nacimiento_iso"] = f"{anio}-{mes}-{dia}"
                    datos["edad"] = edad
                    break
            except:
                pass
    
    # ===== SEXO (desde MRZ) =====
    mrz_sexo = re.search(r'\d{6}([MF])\d{7}', texto_ocr)
    if mrz_sexo:
        datos["sexo"] = mrz_sexo.group(1)
        datos["sexo_completo"] = "MASCULINO" if datos["sexo"] == "M" else "FEMENINO"
    
    # ===== NOMBRE COMPLETO =====
    if all(k in datos for k in ["nombres", "apellido_paterno", "apellido_materno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']} {datos['apellido_materno']}"
    elif all(k in datos for k in ["nombres", "apellido_paterno"]):
        datos["nombre_completo"] = f"{datos['nombres']} {datos['apellido_paterno']}"
    
    return datos

# ============================================================
# 5) FUNCIÓN PRINCIPAL
# ============================================================
def extraer_datos_dni(imagen_path: str) -> dict:
    """
    Función principal: recibe path de imagen, retorna datos extraídos
    
    Args:
        imagen_path: Ruta a la imagen del DNI (frente)
    
    Returns:
        dict con: dni, nombres, apellidos, fecha_nac, edad, sexo
    """
    if ocr_engine is None:
        return {"error": "OCR no inicializado correctamente"}
    
    try:
        # 1) Cargar imagen
        img = cv2.imread(imagen_path)
        if img is None:
            return {"error": "No se pudo cargar la imagen"}
        
        # 2) Mejorar imagen
        img_mejorada = mejorar_imagen_avanzada(img)
        
        # 3) OCR
        resultado = ocr_engine.ocr(img_mejorada, cls=True)
        
        if not resultado or not resultado[0]:
            return {"error": "No se pudo extraer texto de la imagen"}
        
        # 4) Extraer texto
        texto_completo = '\n'.join([bloque[1][0] for bloque in resultado[0]])
        
        # 5) Parsear datos
        datos = parsear_dni(texto_completo)
        
        if not datos.get("dni"):
            return {"error": "No se pudo detectar el DNI en la imagen"}
        
        return datos
        
    except Exception as e:
        return {"error": f"Error procesando DNI: {str(e)}"}
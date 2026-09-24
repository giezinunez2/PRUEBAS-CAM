"""
Launcher del Simulador ADAS 3D.
Permite ejecutar cualquiera de los 6 niveles de prueba del proyecto
sin tener que recordar el nombre exacto del script ni abrir 2 terminales a mano.
"""
import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NIVELES = {
    "1": ("Nivel 0 - Emisor bus eCAL (sin video)", ["send_message.py"]),
    "2": ("Nivel 0 - Receptor bus eCAL (sin video)", ["receive_message.py"]),
    "3": ("Nivel 1 - Video 1 cámara (emisor)", ["webcam_send.py"]),
    "4": ("Nivel 1 - Video 1 cámara (receptor)", ["webcam_receive.py"]),
    "5": ("Nivel 2 - Video 1 cámara + IA (receptor)", ["webcam_receive_ia.py"]),
    "6": ("Nivel 3 - Video dual cámara (emisor)", ["Send_tripleVideo.py"]),
    "7": ("Nivel 3 - Video dual cámara (receptor, sin IA)", ["Receive_tripleVideo.py"]),
    "8": ("Nivel 4 - Dashboard ADAS completo (receptor con IA + BEV)", ["byd_adas_ecal.py"]),
    "9": ("Nivel 5 - Simulación standalone (sin eCAL)", ["byd_adas_simulation.py"]),
}

def run_script(nombre_archivo):
    path = os.path.join(BASE_DIR, nombre_archivo)
    if not os.path.exists(path):
        print(f"[ERROR] No se encontró {path}")
        return
    subprocess.run([sys.executable, path])

def print_menu():
    print("\n=== Simulador ADAS 3D - Launcher ===")
    for key, (desc, _) in NIVELES.items():
        print(f"  {key}. {desc}")
    print("  0. Salir")

def main():
    while True:
        print_menu()
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "0":
            print("Saliendo...")
            break

        if opcion in NIVELES:
            desc, scripts = NIVELES[opcion]
            print(f"\nEjecutando: {desc}")
            print("(Recuerda: si es un par emisor/receptor, necesitas correr el otro script en otra terminal)")
            for script in scripts:
                run_script(script)
        else:
            print("Opción inválida, intenta de nuevo.")

if __name__ == "__main__":
    main()
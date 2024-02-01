import random
import sqlite3

def generar_censo(cantidad):
  censo = []
  alfabeto = "ABCDFDEDASDFGEQRWETYUIOPLKJDHMMAEAAOIOUE"
  numero = 0
  
  for i in range(cantidad):
      aumento = random.randint(1, 2)
      numero += aumento

      letras = random.sample(alfabeto, 5)
      nombre = "".join(letras)

      edad = random.randint(18, 99)
      impuestos = random.choice((True, True, True, False))

      censo.append([numero, nombre, edad, impuestos])

  return censo

def main():
  cantidad_registros = 1050000
  censo = generar_censo(cantidad_registros)

  # Conexion a la base de datos
  conn = sqlite3.connect("censo.db") # Si no se tiene la base de datos, se crea
  c = conn.cursor() # Se crea un cursor para ejecutar las sentencias SQL

  # Crear una tabla si no existe
  c.execute('''CREATE TABLE IF NOT EXISTS censo (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              numero INTEGER,
              nombre TEXT,
              edad INTEGER,
              impuestos BOOLEAN)''')
  # Insertar los registros en la tabla
  c.executemany('INSERT INTO censo (numero, nombre, edad, impuestos) VALUES (?, ?, ?, ?)', censo)
  conn.commit() # Guardar los cambios en la base de datos
  conn.close() # Cerrar la conexión a la base de datos

  print("Censo creado con exito")

if __name__ == "__main__":
  main()
                
  
import sqlite3

def leer_datos():
  conn = sqlite3.connect("censo.db")
  cursor = conn.cursor()

  cursor.execute('SELECT * FROM censo LIMIT 10;')

  # Obtener y mostrar los datos
  rows = cursor.fetchall()
  for row in rows:
    print(row)  # Aquí puedes ajustar la forma en que se muestran los datos

  # Cerrar la conexión a la base de datos
  conn.close()

if __name__ == '__main__':
    leer_datos()
from flask import Flask, request, render_template_string # Importar Flask para crear la aplicación web,  y request para manejar solicitudes HTTP y render_template_string para renderizar plantillas
import sqlite3

app = Flask(__name__) # Crear una aplicación de Flask

# Función para conectarse a la base de datos SQLite
def get_db_connection():
    conn = sqlite3.connect('censo.db')
    conn.row_factory = sqlite3.Row # Para poder acceder a las columnas por nombre en lugar de índice
    return conn

# Ruta para la página de inicio
@app.route('/') # Decorador para indicar la ruta de la página
def index():
    """
    Página de inicio con un formulario para realizar búsquedas en el censo.
    """
    return render_template_string('''
        <h1>Búsqueda en el Censo</h1>
        <form action="/buscar" method="post">
            <label for="tipo">Buscar por:</label>
            <select name="tipo" id="tipo">
                <option value="numero">Número</option>
                <option value="nombre">Nombre</option>
            </select>
            <label for="valor">Valor:</label>
            <input type="text" name="valor" id="valor" required>
            <button type="submit">Buscar</button>
        </form>
    ''')

# Ruta para realizar la búsqueda
@app.route('/buscar', methods=['POST'])
def buscar():
    print("Solicitud POST recibida")
    tipo = request.form['tipo']
    valor = request.form['valor']
    print(f"Tipo de busqueda: {tipo}")
    print(f"Valor de busqueda: {valor}")
    conn = get_db_connection()
    registro = None

    if tipo == 'numero': # Buscar por número
        registro = conn.execute('SELECT * FROM censo WHERE numero = ?', (valor,)).fetchone()
    elif tipo == 'nombre': # Buscar por nombre
        registro = conn.execute('SELECT * FROM censo WHERE nombre = ?', (valor,)).fetchone()
    conn.close()
    
    if registro:
      
        return render_template_string('''
            <p>Número: {{ registro['numero'] }}</p>
            <p>Nombre: {{ registro['nombre'] }}</p>
            <p>Edad: {{ registro['edad'] }}</p>
            <p>Impuestos: {{ registro['impuestos'] }}</p>
            <a href="/">Volver</a>
        ''', registro=registro)
    else:
        return 'Registro no encontrado. <a href="/">Volver</a>'

if __name__ == '__main__':
    app.run('0.0.0.0', port=8080)

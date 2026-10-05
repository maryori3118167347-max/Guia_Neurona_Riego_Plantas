import numpy as np

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

# La humedad se encuentra entre 0 y 100.
# La temperatura se encuentra aproximadamente de 0 y 50.
# Dividimos cada columna por su valor máximo esperado
# para dejar los datos en escala cercana a 0 y 1.
escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def entrenar(tasa_aprendizaje, epocas, mostrar=True):
    generador = np.random.default_rng(42)
    pesos = generador.uniform(-1, 1, size=(2, 1))
    sesgo = generador.uniform(-1, 1)

    for epoca in range(1, epocas + 1):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)

        error = probabilidad - y
        ecm = np.mean(error ** 2)

        gradiente_z = error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z / len(X)
        gradiente_sesgo = np.sum(gradiente_z) / len(X)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and epoca % 1000 == 0:
            print(f"Época {epoca}: error = {ecm:.6f}")

    return pesos, sesgo

tasa_aprendizaje = 0.5
epocas = 10000

pesos, sesgo = entrenar(tasa_aprendizaje, epocas)

z = X_normalizado @ pesos + sesgo
probabilidad = sigmoide(z)
respuesta = (probabilidad >= 0.5).astype(int)
error_final = np.mean((probabilidad - y) ** 2)

print("\nResultados del entrenamiento:")
print(f"Peso de la humedad: {pesos[0, 0]:.4f}")
print(f"Peso de la temperatura: {pesos[1, 0]:.4f}")
print(f"Sesgo: {sesgo:.4f}")
print(f"Error final: {error_final:.6f}")

print("\nProbabilidad de cada caso:")
for i in range(len(X)):
    print(f"Humedad {X[i, 0]:.0f} %, temperatura {X[i, 1]:.0f} °C: "
          f"probabilidad {probabilidad[i, 0]:.4f}, respuesta {respuesta[i, 0]}, "
          f"esperado {y[i, 0]:.0f}")

print(f"Respuestas correctas: {np.sum(respuesta == y)} de {len(X)}")


nuevos = np.array([
    [75, 30], [45, 34], [25, 22], [50, 25], [30, 40]
], dtype=float)

nuevos_normalizados = nuevos / escala

prob_nuevos = sigmoide(nuevos_normalizados @ pesos + sesgo)
resp_nuevos = (prob_nuevos >= 0.5).astype(int)

print("\nPruebas con condiciones nuevas:")
for i in range(len(nuevos)):
    print(f"Humedad {nuevos[i, 0]:.0f} %, temperatura {nuevos[i, 1]:.0f} °C: "
          f"probabilidad {prob_nuevos[i, 0]:.4f}, decisión {resp_nuevos[i, 0]}")


# Experimentos: en cada prueba cambia un solo parámetro respecto a la base.
experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

caso_prueba = np.array([[45, 34]], dtype=float) / escala

print("\nExperimentos con los parámetros:")
for nombre, ep, tasa in experimentos:
    p, s = entrenar(tasa, ep, mostrar=False)

    prob = sigmoide(X_normalizado @ p + s)
    error_exp = np.mean((prob - y) ** 2)
    correctas = np.sum((prob >= 0.5).astype(int) == y)
    prob_caso = sigmoide(caso_prueba @ p + s)[0, 0]

    print(f"{nombre}: épocas {ep}, tasa {tasa}, error {error_exp:.6f}, "
          f"correctas {correctas}/10, P(45 %, 34 °C) = {prob_caso:.4f}")


# Umbral: se usan los pesos y el sesgo del entrenamiento base, sin reentrenar.
todos = np.vstack([X, nuevos])
prob_todos = sigmoide(todos / escala @ pesos + sesgo)

print("\nPrueba con distintos umbrales (sin volver a entrenar):")
for i in range(len(todos)):
    r4 = int(prob_todos[i, 0] >= 0.4)
    r5 = int(prob_todos[i, 0] >= 0.5)
    r6 = int(prob_todos[i, 0] >= 0.6)
    print(f"Humedad {todos[i, 0]:.0f} %, temperatura {todos[i, 1]:.0f} °C: "
          f"probabilidad {prob_todos[i, 0]:.4f} -> "
          f"umbral 0.4: {r4}, umbral 0.5: {r5}, umbral 0.6: {r6}")

print(f"\nPesos y sesgo después de cambiar el umbral: "
      f"{pesos[0, 0]:.4f}, {pesos[1, 0]:.4f}, {sesgo:.4f}")
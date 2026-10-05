Neurona artificial para el riego de plantas



Neurona artificial hecha con Python y NumPy que decide si una planta necesita riego a partir de dos entradas: la humedad del suelo y la temperatura ambiental. Usa la función de activación sigmoide y entrega una salida binaria: 1 = regar, 0 = no regar.

Los datos son didácticos y no representan una recomendación agronómica para una especie real.


Descripción del problema



| Variable | Significado | Escala |

|---|---|---|

| x1 | Humedad del suelo | 0 a 100 % |

| x2 | Temperatura ambiental | 0 a 50 °C aprox. |

| y | Decisión | 1 = regar, 0 = no regar |



Datos de entrenamiento:

 Caso  Humedad , Temperatura  Salida 


 1  80 % , 18 °C     0

 2  70 % , 22 °C     0

 3  65 % ,28 °C      0 

 4  55 % , 25 °C     0 

 5  50 % , 32 °C     0 

 6  40 % , 30 °C     1 

 7  35 % , 25 °C     1 

 8  30 % , 32 °C     1 

 9  20 % , 35 °C     1 

 10  10 % , 38 °C    1 



Cómo ejecutarlo



Se necesita tener instalado \[uv](https://docs.astral.sh/uv/).



```bash

git clone https://github.com/maryori3118167347-max/Guia\_Neurona\_Riego\_Plantas.git

cd Guia\_Neurona\_Riego\_Plantas

uv sync

uv run main.py

```



\## Cómo funciona la neurona



1. Normalización:
cada columna se divide por su valor máximo esperado, `escala = \[100, 50]`.

2.Suma ponderada:
z = X\_normalizado @ pesos + sesgo` (dos pesos y un sesgo).

3. Activación:
sigmoide(z) = 1 / (1 + e^(-z))`, que convierte `z` en una probabilidad entre 0 y 1.

4.Error:
error cuadrático medio entre la probabilidad y la salida esperada.

5. Gradientes y actualización:
se calculan los gradientes y se ajustan los pesos y el sesgo en cada época.

6. Decisión:
respuesta = (probabilidad >= 0.5).astype(int)`.



Los pesos iniciales se generan con una semilla fija (`42`), así todos los experimentos parten del mismo punto y lo único que cambia entre uno y otro es el parámetro que se está probando.


Resultados del entrenamiento base


Configuración: `tasa\_aprendizaje = 0.5`, `epocas = 10000`.



Parámetro  Valor 


 Peso de la humedad  -14.9497 

 Peso de la temperatura  2.6066 

 Sesgo  5.2147 

 Error final (ECM)  0.030334 


 Caso  Humedad  Temperatura  Esperado  Probabilidad  Respuesta 


| 1  80 %  18 °C | 0    0.0030 | 0 

| 2  70 %  22 °C | 0    0.0163 | 0 

| 3  65 %  28 °C | 0    0.0455 | 0 

| 4  55 %  25 °C | 0    0.1539 | 0 

| 5  50 %  32 °C | 0    0.3562 | 0 

| 6  40 %  30 °C | 1    0.6897 | 1 

| 7  35 %  25 °C | 1    0.7834 | 1 

| 8  30 %  32 °C | 1    0.9167 | 1 

| 9  20 %  35 °C | 1    0.9829 | 1 

|10  10 %  38 °C | 1    0.9967 | 1 



Respuestas correctas: \*\*10 de 10\*\*.



Signo de los pesos.
El peso de la humedad es negativo: cuanto más húmedo está el suelo, más baja `z` y menor es la probabilidad de regar. El peso de la temperatura es positivo: cuanto más calor hace, mayor es la probabilidad de regar. Además, el peso de la humedad es mucho más grande en valor absoluto (14.95 frente a 2.61), lo que indica que la neurona decide sobre todo por la humedad y usa la temperatura como un ajuste menor.



Predicciones con condiciones nuevas



Los cinco datos se normalizan con la misma variable `escala` usada en el entrenamiento.



| Humedad | Temperatura | Probabilidad | Decisión |


| 75 % | 30 °C | 0.0117 | 0 (No regar) |

| 45 % | 34 °C | 0.5646 | 1 (Regar) |

| 25 % | 22 °C | 0.9324 | 1 (Regar) |

| 50 % | 25 °C | 0.2775 | 0 (No regar) |

| 30 % | 40 °C | 0.9435 | 1 (Regar) |



El caso de 45 % y 34 °C es el más dudoso: su probabilidad queda muy cerca de 0.5 porque está justo en la frontera entre los ejemplos de "regar" y "no regar".



\## Experimentos con los parámetros



En cada prueba se cambió un solo parámetro respecto a la prueba base.



| Experimento | Épocas | Tasa | Error final | Correctas | P(45 %, 34 °C) | Aprendizaje |


| Prueba base | 10000 | 0.5 | 0.030334 | 10/10 | 0.5646 | Adecuado |

| Pocas épocas | 100 | 0.5 | 0.216162 | 10/10 | 0.4914 | Insuficiente |

| Cantidad intermedia | 1000 | 0.5 | 0.089950 | 10/10 | 0.5521 | Incompleto |

| Más épocas | 20000 | 0.5 | 0.019455 | 10/10 | 0.5442 | Adecuado, mejora pequeña |

| Tasa pequeña | 10000 | 0.01 | 0.177776 | 10/10 | 0.5046 | Lento |

| Tasa moderada | 10000 | 0.1 | 0.066760 | 10/10 | 0.5688 | Algo lento |

| Tasa alta | 10000 | 1.0 | 0.019454 | 10/10 | 0.5442 | Rápido |

| Tasa muy alta | 10000 | 2.0 | 0.011639 | 10/10 | 0.5231 | Rápido, sin inestabilidad |



Observaciones:



Todos los experimentos aciertan los 10 casos, pero eso engaña. Con 100 épocas el error sigue siendo alto (0.216): la neurona acierta por muy poco, con probabilidades cercanas a 0.5. El conteo de aciertos no alcanza para evaluar el entrenamiento; hay que mirar también el error.

Con 100 épocas cambia una decisión real:para 45 % y 34 °C la probabilidad es 0.4914, es decir \*no regar\*, mientras que en todos los demás experimentos da \*regar\*.

Duplicar la tasa equivale casi a duplicar las épocas: tasa 1.0 con 10000 épocas (0.019454) da prácticamente el mismo error que tasa 0.5 con 20000 épocas (0.019455).

No apareció inestabilidad. Ni siquiera con tasa 2.0, que fue la que logró el menor error. En este problema las tasas altas simplemente aprendieron más rápido.



 Prueba adicional con el umbral



Se usaron los pesos y el sesgo del entrenamiento base, sin volver a entrenar.



| Humedad | Temperatura | Probabilidad | Umbral 0.4 | Umbral 0.5 | Umbral 0.6 |



| 80 % | 18 °C | 0.0030 | 0 | 0 | 0 |

| 70 % | 22 °C | 0.0163 | 0 | 0 | 0 |

| 65 % | 28 °C | 0.0455 | 0 | 0 | 0 |

| 55 % | 25 °C | 0.1539 | 0 | 0 | 0 |

| 50 % | 32 °C | 0.3562 | 0 | 0 | 0 |

| 40 % | 30 °C | 0.6897 | 1 | 1 | 1 |

| 35 % | 25 °C | 0.7834 | 1 | 1 | 1 |

| 30 % | 32 °C | 0.9167 | 1 | 1 | 1 |

| 20 % | 35 °C | 0.9829 | 1 | 1 | 1 |

| 10 % | 38 °C | 0.9967 | 1 | 1 | 1 |

| 75 % | 30 °C (nuevo) | 0.0117 | 0 | 0 | 0 |

| \*\*45 %\*\* | \*\*34 °C (nuevo)\*\* | \*\*0.5646\*\* | \*\*1\*\* | \*\*1\*\* | \*\*0\*\* |

| 25 % | 22 °C (nuevo) | 0.9324 | 1 | 1 | 1 |

| 50 % | 25 °C (nuevo) | 0.2775 | 0 | 0 | 0 |

| 30 % | 40 °C (nuevo) | 0.9435 | 1 | 1 | 1 |



El único caso que cambia es 45 % y 34 °C=
su probabilidad (0.5646) supera 0.4 y 0.5, pero no llega a 0.6, así que con el umbral más exigente pasa de \*regar\* a \*no regar\*. Ninguno de los diez casos de entrenamiento cambia, porque ninguna de sus probabilidades cae entre 0.4 y 0.6 (la más cercana es 0.3562).



Modificar el umbral no cambia los pesos porque el umbral no participa en el entrenamiento. Los pesos y el sesgo se ajustan con el error entre la probabilidad y la salida esperada; el umbral se aplica después, solo para traducir esa probabilidad en una decisión. Tras la prueba los valores siguen siendo -14.9497, 2.6066 y 5.2147.



 Análisis



1. ¿Por qué fue necesario normalizar la humedad y la temperatura?

Porque están en escalas distintas: la humedad llega a 100 y la temperatura a unos 50. Sin normalizar, la humedad pesaría más solo por tener números más grandes, y valores como 80 producirían una `z` enorme que satura la sigmoide (salida pegada a 0 o a 1), donde el gradiente es casi cero y la neurona casi no aprende. Al dividir por `\[100, 50]` las dos entradas quedan entre 0 y 1 y son comparables.



2. ¿En qué operaciones se utilizó `X\_normalizado` y para qué se conservó `X`?

`X\_normalizado` se usó en la suma ponderada (`z = X\_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X\_normalizado.T @ gradiente\_z`). `X` se conservó para mostrar los datos en sus unidades originales (% y °C), que es como se entienden los resultados.



3. ¿Qué ocurrió al utilizar solamente 100 épocas?

El entrenamiento fue insuficiente. El error quedó en 0.216, siete veces el de la prueba base. La neurona clasificó bien los diez casos, pero con muy poca seguridad, y además cambió la decisión para 45 % y 34 °C a \*no regar\* (probabilidad 0.4914).



4. ¿Más épocas siempre produjeron una mejora importante?

No. De 100 a 1000 épocas el error bajó mucho (0.216 a 0.090) y de 1000 a 10000 también (a 0.030), pero de 10000 a 20000 solo bajó a 0.019 y no cambió ninguna decisión. Cada vez se gana menos: después de cierto punto, más épocas significan más tiempo de cálculo para una mejora pequeña.



5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?

Con 0.01 el aprendizaje fue muy lento: después de 10000 épocas el error todavía era 0.178, peor que el de la tasa 0.5 con solo 1000 épocas. Los pasos son tan pequeños que la neurona no alcanza a llegar a una buena solución en las épocas disponibles.



6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?

En este problema, aprendió más rápido: con 1.0 el error fue 0.0195 y con 2.0 fue 0.0116, el más bajo de todos. No se observó inestabilidad. En general una tasa muy alta puede hacer que el error oscile o se dispare porque los pasos se pasan del mínimo, pero aquí no ocurrió, probablemente porque los datos están normalizados y el problema es sencillo.



7. ¿Qué representa el signo del peso correspondiente a la humedad?

Es negativo (-14.95): a mayor humedad del suelo, menor probabilidad de regar. Tiene sentido, porque un suelo húmedo no necesita agua.



8. ¿Qué representa el signo del peso correspondiente a la temperatura?

Es positivo (2.61): a mayor temperatura, mayor probabilidad de regar, porque el calor seca más rápido el suelo. Su valor es mucho menor que el de la humedad, así que influye menos en la decisión.



9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?

Porque la sigmoide entrega un valor continuo entre 0 y 1, y la acción real es binaria: se riega o no se riega. El umbral define desde qué nivel de probabilidad se toma la decisión. Un umbral bajo riega con más facilidad; uno alto exige más seguridad antes de regar.



10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?

Solo tiene diez datos didácticos y se evalúa con los mismos datos con los que se entrenó, así que no se sabe qué tan bien generaliza.

 Solo considera dos variables. No tiene en cuenta la especie, el tipo de suelo, la lluvia, la luz, el viento, el tamaño de la maceta ni la hora del día.

 Una sola neurona solo puede separar los casos con una línea recta. No puede aprender reglas más complejas.

 No dice cuánta agua aplicar, solo si regar o no.

La escala está fijada a mano (100 y 50); una temperatura por encima de 50 °C quedaría fuera del rango esperado.


import numpy as np
import sympy as sp
#Primera parte-matrices
matriz_A= np.array([[1,2,0],[0,1,1],[3,0,2]])
matriz_B= np.array([[2,1,1],[1,0,3],[0,2,1]])

multiplica = np.multiply(matriz_A, matriz_B)
bT= matriz_B.T
#Segunda parte -Vectores
vector_u = np.array([1,-2,3])
vector_v = np.array([0,1,4])

producto_interno = np.dot(vector_u, vector_v)
#magnitudes
magnitud_u = np.linalg.norm(vector_u)
magnitud_v = np.linalg.norm(vector_v)
#angulo
cos_rad = np.arccos(producto_interno / (magnitud_u * magnitud_v))
angulo= np.degrees(cos_rad)

#Tercera parte- Derivadas
# f(x) = x**4 - 3*x**3 + 2*x - 5 
x = sp.symbols('x')
f = x**4 - 3*x**3 + 2*x - 5
f1 = sp.diff(f, x)
f2= sp.diff(f, x, 2)
#evaluar en x=2 y comentar concavidad (el comentario se encuentra en los prints)
f1_eval = f1.subs(x, 2)
f2_eval = f2.subs(x, 2)



print(f"La multiplicacion de A y B:\n {multiplica}")
print(f"Traspuesta de B:\n {bT} ")
print(f"Producto interno de u y v:\n {producto_interno}")
print(f"Magnitud de u:\n {magnitud_u}")
print(f"Magnitud de v:\n {magnitud_v}")
print(f"Angulo en radianes:\n {angulo}")
print(f"Primera derivada de f(x): {f1}")
print(f"Segunda derivada de f(x): {f2}")
# Comentario sobre concavidad
# Viendo que el resultado de la segunda derivada es 12 positivo, la función es concava hacia arriba en x=2
#Y ya que la primera derivada en x=2 es -2, la función está decreciendo en ese punto.
# igual ya se hizo un if para que comente eso.
print(f"Primera derivada en x=2: {f1_eval}")
print(f"Segunda derivada de f(x): {f2_eval}")
if f2_eval > 0:
    print("La funcion es concava hacia arriba en x=2.")
elif f2_eval < 0:
    print("La funcion es concava hacia abajo en x=2.")
if f1_eval > 0:
    print("La funcion esta creciendo en x=2.")
elif f1_eval < 0:
    print("La funcion esta decreciendo en x=2.")
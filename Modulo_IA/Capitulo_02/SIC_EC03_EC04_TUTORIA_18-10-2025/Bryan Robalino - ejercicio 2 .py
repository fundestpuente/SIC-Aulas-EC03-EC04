# Taller en clase:
import numpy as np
import math
import sympy as sp



"""
----------------------------------------------------------
EJERCICIO 2
----------------------------------------------------------
"""

# Ejercicio 2
print("==="*50)
print("Matrices")
print("==="*50)
# Matrices:
A = [[1, 0, 2], [2, 1, 1], [3, 2, 0]]
B = [[0, 1, 1], [1, 2, 0], [2, 1, 3]]

# Mostrar las matrices 
print("Matriz A: \n", A)
print("Matriz B: \n", B)

# Calcula A × B.
# Multiplicacion de matrices

print("A x B = \n", np.dot(A, B))

# Calcula la traspuesta y determinante de A.
# Traspuesta de A
traspuesta_A = np.transpose(A)
print("\nTraspuesta de A (A^T) =\n", traspuesta_A)

# Determinante de A
det_A = np.linalg.det(A)
print(f"\nDeterminante de A: {det_A:.2f}")


# Vectores:
u = [2, -1, 3]
v = [1, 4, -2]

print("==="*50)
print("Vectores")
print("==="*50)
# Calcula el producto interno u·v y determina si son ortogonales.
producto_interno = np.dot(u, v)
print(f"Producto interno (u·v): {producto_interno}")

# Verificar si son ortogonales
if producto_interno == 0:
    print("Los vectores son ortogonales (perpendiculares)")
else:
    print("Los vectores no son ortogonales")
# Calcula el ángulo entre u y v.

# Fórmula: cos(θ) = (u·v) / (|u| * |v|)
mod_u = np.linalg.norm(u)
mod_v = np.linalg.norm(v)

cos_theta = producto_interno / (mod_u * mod_v)

# Convertir a grados
angulo = math.degrees(math.acos(cos_theta))

print(f"Ángulo entre u y v: {angulo:.2f} grados")

# Derivadas:
print("==="*50)
print("Derivadas")
print("==="*50)
# g(x, y) = x*2 * y + sin(xy)
x, y = sp.symbols('x y')
g = x**2 * y + sp.sin(x * y)
print("Función g(x, y) =", g)

# Calcula las derivadas parciales ∂g/∂x y ∂g/∂y.

dg_dx = sp.diff(g, x)  # ∂g/∂x
dg_dy = sp.diff(g, y)  # ∂g/∂y

print("\n∂g/∂x =", dg_dx)
print("∂g/∂y =", dg_dy)

# Evalúa en el punto (x=1, y=π/2).

valor_x = 1
valor_y = math.pi / 2

dg_dx_eval = dg_dx.subs({x: valor_x, y: valor_y})
dg_dy_eval = dg_dy.subs({x: valor_x, y: valor_y})

print(f"\nEn el punto (x=1, y=π/2):")
print(f"∂g/∂x = {dg_dx_eval.evalf():.4f}")
print(f"∂g/∂y = {dg_dy_eval.evalf():.4f}")
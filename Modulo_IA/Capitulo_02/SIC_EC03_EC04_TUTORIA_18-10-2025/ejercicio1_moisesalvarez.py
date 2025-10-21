import numpy as np
import sympy as sp
A = np.array([[1, 2], 
              [3, 4]])

B = np.array([[2, 0], 
              [1, 2]])

print(f"A:\n{A}\n")
print(f"B:\n{B}\n")

suma_A_B = A + B
print(f"A + B:\n{suma_A_B}\n")

resta_A_B = A - B
print(f"A - B:\n{resta_A_B}\n")

A_x_B = np.dot(A, B)
print(f"Producto Matricial (A x B):\n{A_x_B}\n")

A_transpuesta = A.T  
print(f"Traspuesta de A :\n{A_transpuesta}\n")

try:
    B_inversa = np.linalg.inv(B)
    print(f"Inversa de B (B^-1):\n{B_inversa}\n")
except np.linalg.LinAlgError:
    print("La matriz B no tiene inversa\n")


# -----------------------------------------------
u = np.array([1, 2])
v = np.array([3, -1])

print(f"u: {u}")
print(f"v: {v}\n")

producto_interno =np.dot(u, v)
print(f"Producto interno u.v {producto_interno}\n")

mag_u = np.linalg.norm(u)
mag_v = np.linalg.norm(v)

print(f"Magnitud de u (|u|): {mag_u}")
print(f"Magnitud de v (|v|): {mag_v}\n")


# -----------------------------------------------


x = sp.symbols('x')
f = x**3 + 2*x**2 + x
print(f" f(x) = {f}\n")

fd= sp.diff(f, x)
print(f"Primera derivada f'(x) = {fd}\n")

fd2 = sp.diff(f, x, 2) 
print(f"Segunda derivada f''(x) = {fd2}\n")

fd_eval_en_1 = fd.subs(x, 1)
f2_eval_en_1 = fd2.subs(x, 1)

print(f"f'(1) = {fd_eval_en_1}")
print(f"f''(1) = {f2_eval_en_1}")
import sympy as sp
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

print("Matrices")
print("")
print("A =\n", A)
print("B =\n", B)
print("\nA + B =\n", A + B)
print("\nA - B =\n", A - B)
print("\nA × B =\n", A @ B)
print("\nTranspuesta de A =\n", A.T)

detB = np.linalg.det(B)
print("\nDeterminante de B =", detB)

if detB != 0:
    print("\nInversa de B =\n", np.linalg.inv(B))
else:
    print("La matriz B no tiene inversa")

u = np.array([1, 2])
v = np.array([3, -1])

print("")
print("Vectores")
print("")
print("u =", u)
print("v =", v)
print("\nProducto interno (u·v) =", np.dot(u, v))
print("|u| =", np.linalg.norm(u))
print("|v| =", np.linalg.norm(v))


x = sp.symbols('x')
f = x**3 + 2*x**2 + x
f1 = sp.diff(f, x)
f2 = sp.diff(f1, x)

print("derivadas")
print("")
print("f(x) =", f)
print("f'(x) =", f1)
print("f''(x) =", f2)
print("\nf'(1) =", f1.subs(x, 1))
print("f''(1) =", f2.subs(x, 1))


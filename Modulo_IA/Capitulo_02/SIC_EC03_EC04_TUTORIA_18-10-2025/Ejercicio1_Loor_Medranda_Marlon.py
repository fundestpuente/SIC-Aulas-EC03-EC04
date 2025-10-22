import numpy as np
import sympy as sp

# Alumno: Loor Medranda Marlon
# Ejercicio 1 — Operaciones con matrices, vectores y derivadas

def main():
    # --- Matrices ---
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[2, 0],
                  [1, 2]])

    print("ALUMNO: Loor Medranda Marlon\n")
    print("MATRICES")
    print("Matriz A:\n", A)
    print("Matriz B:\n", B)

    suma_AB = A + B
    resta_AB = A - B
    producto_AB = A.dot(B)  # producto matricial
    transpuesta_A = A.T

    det_B = np.linalg.det(B)
    inversa_B = None
    if not np.isclose(det_B, 0):
        inversa_B = np.linalg.inv(B)

    print("\nA + B =\n", suma_AB)
    print("\nA - B =\n", resta_AB)
    print("\nA x B =\n", producto_AB)
    print("\nTranspuesta de A (A^T) =\n", transpuesta_A)
    print("\nDeterminante de B =", det_B)
    if inversa_B is not None:
        print("\nInversa de B (B^-1) =\n", inversa_B)
    else:
        print("\nLa matriz B no tiene inversa (determinante = 0)")

    # --- Vectores ---
    u = np.array([1, 2])
    v = np.array([3, -1])

    producto_interno = np.dot(u, v)
    norma_u = np.linalg.norm(u)
    norma_v = np.linalg.norm(v)

    print("\n\nVECTORES")
    print("u =", u)
    print("v =", v)
    print("\nProducto interno u · v =", producto_interno)
    if np.isclose(producto_interno, 0):
        print("Los vectores u y v son ortogonales.")
    else:
        print("Los vectores u y v NO son ortogonales.")

    print("\n|u| =", norma_u)
    print("|v| =", norma_v)

    # --- Derivadas ---
    x = sp.symbols('x')
    f = x**3 + 2*x**2 + x

    f1 = sp.diff(f, x)
    f2 = sp.diff(f, x, 2)

    f1_at_1 = f1.subs(x, 1)
    f2_at_1 = f2.subs(x, 1)

    print("\n\nDERIVADAS")
    print("f(x) =", f)
    print("f'(x) =", f1)
    print("f''(x) =", f2)
    print("\nf'(1) =", f1_at_1)
    print("f''(1) =", f2_at_1)


if __name__ == "__main__":
    main()
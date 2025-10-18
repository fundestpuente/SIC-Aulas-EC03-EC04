## Ejercicio 1
import numpy as np 
import sympy as sp
#**Matrices:**  
A = np.array([[1, 2], [3, 4]] )
B = np.array([[2, 0], [1, 2]] )

#1. Calcula A + B, A - B y el producto matricial A × B.  
print("A + B =", np.add(A,B))
print("A - B =", np.subtract(A,B))
print("A X B =", np.dot(A,B))
#2. Calcula la traspuesta de A y la inversa de B (si existe).
print("Transpuesta de A:", A.T)
determinante_B= np.linalg.det(B)
print("Determinante de B:", determinante_B)

if determinante_B !=0:
    inversa_B= np.linalg.inv(B)
    print("Inversa de B:",inversa_B)


#**Vectores:**  
u = np.array([1, 2])
v = np.array([3, -1])

#1. Calcula el producto interno u·v.  
producto_interno= np.dot(u,v)
print("Producto interno", producto_interno)
if producto_interno == 0:
    print("Los vectores son ortogonales perpendiculares")
else:
    print("Los vectores son ortogonales")
    
#2. Calcula la magnitud de cada vector.  
magnitud_u = np.linalg.norm(u)
magnitud_v = np.linalg.norm(v)
print("Magnitud de u:", magnitud_u)
print("Magnitud de v:", magnitud_v)

#**Derivadas:**  
#f(x) = x**3 + 2*x**2 + x  

#1. Calcula la derivada primera y segunda de f(x). 

x= sp.symbols('x') 
f= x**3 + 2*x**2 + x  

df= sp.diff(f,x)
print("Derivada de primer orden", df)
d2f= sp.diff(f,x,2)
print("Derivada de segundo orden", d2f)

#2. Evalúa ambas derivadas en x = 1.
df_valor = df.subs(x, 1)
d2f_valor = d2f.subs(x, 1)
print("Derivada primera en x=1:", df_valor)
print("Derivada segunda en x=1:", d2f_valor)
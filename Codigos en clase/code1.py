"""
tema: classificador ballesiana
objetivo: dado un conjuto de datos, clasificar las observaciones haciendo uso de los
clasificadores bayesianos y naive y concluir acerca de ellos
"""
from sklearn import datasets
# ------------------------------pre procesamiento de datos--------------------------------------
db = datasets.load_iris()

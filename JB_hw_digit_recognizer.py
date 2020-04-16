import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

class HWDigitRecognizer:
    
  def __init__(self, train_filename, test_filename):
    """El método init leerá los datasets con extensión ".csv" cuyas ubicaciones son recibidas mediante los paramentros <train_filename> y <test_filename>. Los usará para crear las matrices de X_train, X_test, Y_train y Y_test con las dimensiones adecuadas y normalización de acuerdo a lo definido en clase.
    """
    self._train_filename = train_filename
    self._test_filename = test_filename

    data_train_filename = pd.read_csv(self._train_filename)
    Y_train = data_train_filename.label
    X_train = data_train_filename.drop('label', axis=1)

    data_test_filename = pd.read_csv(self._test_filename)
    Y_test = data_test_filename.label
    X_test = data_test_filename.drop('label', axis=1)

    #Cantidad de entrenamientos
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]

    # Reshape the training and test examples
    train_set_x_flatten = X_train.reshape(X_train.shape[0], -1).T
    test_set_x_flatten = X_test.reshape(X_test.shape[0], -1).T

    #
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

  def print_test(self):
    #data_test_filename = pd.read_csv(self._test_filename)
    #dt_test=pd.DataFrame(data_test_filename)
    data_train_filename = pd.read_csv(self._train_filename)
    Y_train = data_train_filename.label
    X_train = data_train_filename.drop('label', axis=1)
    return X_train

  def train_model(self):
  # def train_model(self, learn_rate, num_iter):
    """Debido a que el problema a resolver es de clasificaicón multi-clase, este reducirá a un problema de clasificación binaria mediante la técnica de reducción uno-contra-uno (one-vs-one) para crear k(k-1)/2 clasificadores (modelos) que en conjunto resuelven el problema. <k> representa el número de clases diferentes que en el caso de los dígitos es igual a 10.

    Este método entrenará los clasificadores requeridos y retornará un diccionario con los parámetros <w>, <b> y la listas de costos <costs> obtenidos cada 100 interaciones durante el entrenamiento para cada clasificador identificado por un frozenset de la siguiente forma:
    
    params = {
    frozenset({0,1}) : (w,b,costs),
    frozenset({0,2}) : (w,b,costs),
    frozenset({0,3}) : (w,b,costs),
    ...
    frozenset({8,9}) : (w,b,costs),
    "learning_rate" : learning_rate,
    "num_iterations": num_iterations}
    
    Además el diccionario contendrá los valores de <learning_rate> y <num_iterations> usados en el entrenamiento, éstos deben ser definidos por el programador mediante un proceso de prueba y error con el fin de obtener la mejor precisión en los datos de prueba. Por simplicidad, se usarán los mismos valores de <learning_rate> y <num_iterations> para todos los clasificadores creados.

    Finalmente, por razones de eficiencia el autograder revisará su programa usando un dataset más pequeño que el que se proporciona (usted puede hacer lo mismo para sus pruebas iniciales). Pero una vez entregado su proyecto se harán pruebas con el dataset completo, por lo que el diccionario que retorna este método con los resultados del entrenamiento con una precisión mayor al 94% en los datos de prueba debe ser entregado junto con el este archivo completado.

    Para entregar dicho diccionario deberá guardarlo como un archivo usando el módulo "pickle" con el nombre y extensión "params.dict", este archivo deberá estar ubicado en el mismo directorio donde se encuentra  el archivo actual: hw_digit_recognizer.py. El autograder validará que este archivo esté presente y tendra las claves correctas, pero la revisión de la precisión se hará por el docente después de la entrega.
    """
    pass

  def predict(self, X, params, class_pairs):
    """Retorna un vector de <(1,m)> con las etiquetas predecidas para las instancias en X usando la técnica de reducción uno-contra-uno (one-vs-one).

    <class_pairs> contiene una lista con las k(k-1)/2 tuplas que identifican los 
    clasificadores que se utilizarán para la predicción. 

    <params> contiene un diccionario con los parámetros <w> y <b> de cada uno de los clasificadores tal como se explica en la documentación del método <train_model>.

    A cada instancia en <X> se le predice una etiqueta por cada uno de los clasificadores (modelos), es decir cada instancia tendrá inicialmente k(k-1)/2 etiquetas de las cuales se escogerá la que más se repita como etiqueta definitiva para dicha instancia. Si se repiten varias etiquetas un mismo número de veces, se escoge una al azar. Para hacer la selección al azar use el módulo random de la librería numpy, y antes de hacer cada llamado a una función de éste defina la semilla en 1 con <np.random.seed(1)>, esto es para que obtenga resultados "aleatorios" iguales a los que espera el autograder.

    Se asume un umbral de 0.5 para la predicción.
    """
    pass

  def get_datasets(self):
    """Retorna un diccionario con los datasets preprocesados con los datos y 
    dimensiones que se usaron para el entrenamiento
    
    d = { "X_train": X_train,
    "X_test": X_test,
    "Y_train": Y_train,
    "Y_test": Y_test
    }
    """
    pass

  def get_pairs(self):
    """Retorna una lista con las k(k-1)/2 tuplas que identifican los clasificadores que conforman este modelo, ej.: [(0,1), (0,2) ... (8,9)]
    """
    pass

test = HWDigitRecognizer('datasets\mnist_train.csv','datasets\mnist_test.csv')
c= test.print_test()
print (c.shape)
#print (c)

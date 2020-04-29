import pandas as pd
import numpy as np
from itertools import combinations
import pickle

class HWDigitRecognizer:

  def __init__(self, train_filename, test_filename):
    """El método init leerá los datasets con extensión ".csv" 
    cuyas ubicaciones son recibidas mediante los paramentros 
    <train_filename> y <test_filename>. 
    Los usará para crear las matrices de X_train, X_test, Y_train y Y_test 
    con las dimensiones adecuadas y normalización de acuerdo a lo definido en clase.
    """
    comb = combinations([0,1,2,3,4,5,6,7,8,9], 2)
    self.clasificadores = list(comb)
    trainFile = pd.read_csv(train_filename)
    testFile = pd.read_csv(test_filename)

    y_train_len = len(trainFile.label)
    y_test_len = len(testFile.label)

    y_train = (trainFile.label.to_numpy()).reshape(1, y_train_len)
    y_test = (testFile.label.to_numpy()).reshape(1, y_test_len)

    x_train = trainFile.drop('label', axis=1).to_numpy().T
    x_test = testFile.drop('label', axis=1).to_numpy().T

    self.X_train = x_train/255
    self.X_test = x_test/255
    self.Y_train = y_train
    self.Y_test = y_test

  def train_model(self):
  # def train_model(self, learn_rate, num_iter):
    """Debido a que el problema a resolver es de clasificaicón multi-clase,
    este reducirá a un problema de clasificación binaria mediante la técnica 
    de reducción uno-contra-uno (one-vs-one) para crear k(k-1)/2 clasificadores (modelos) 
    que en conjunto resuelven el problema. <k> representa el número de clases diferentes 
    que en el caso de los dígitos es igual a 10.

    Este método entrenará los clasificadores requeridos y retornará un diccionario con 
    los parámetros <w>, <b> y la listas de costos <costs> obtenidos cada 100 
    interaciones durante el entrenamiento para cada clasificador identificado por un 
    frozenset de la siguiente forma:
    
    params = {
    frozenset({0,1}) : (w,b,costs),
    frozenset({0,2}) : (w,b,costs),
    frozenset({0,3}) : (w,b,costs),
    ...
    frozenset({8,9}) : (w,b,costs),
    "learning_rate" : learning_rate,
    "num_iterations": num_iterations}
    
    Además el diccionario contendrá los valores de <learning_rate> y <num_iterations> 
    usados en el entrenamiento, éstos deben ser definidos por el programador mediante 
    un proceso de prueba y error con el fin de obtener la mejor precisión en los datos de 
    prueba. Por simplicidad, se usarán los mismos valores de <learning_rate> y 
    <num_iterations> para todos los clasificadores creados.

    Finalmente, por razones de eficiencia el autograder revisará su programa usando un 
    dataset más pequeño que el que se proporciona (usted puede hacer lo mismo para sus 
    pruebas iniciales). Pero una vez entregado su proyecto se harán pruebas con el dataset 
    completo, por lo que el diccionario que retorna este método con los resultados del 
    entrenamiento con una precisión mayor al 94% en los datos de prueba debe ser entregado 
    junto con el este archivo completado.

    Para entregar dicho diccionario deberá guardarlo como un archivo usando el módulo 
    "pickle" con el nombre y extensión "params.dict", este archivo deberá estar ubicado 
    en el mismo directorio donde se encuentra  el archivo actual: hw_digit_recognizer.py. 
    El autograder validará que este archivo esté presente y tendra las claves correctas, 
    pero la revisión de la precisión se hará por el docente después de la entrega.
    """
    # initialize parameters with zeros (≈ 1 line of code)
    n = self.X_train.shape[0]
    pairs = np.asarray(self.get_pairs())
    print(pairs.shape)
    # print(np.array(self.get_pairs()).shape[0])
    
    dict_params = {}


    for i in range(pairs.shape[0]):      
      Y_tmp = self.Y_train[0]
      X_tmp = self.X_train
      
      Y_tmp = Y_tmp.T
      X_tmp = X_tmp.T

      Y_tmp_train = []
      X_tmp_train = []
      for j in range(len(Y_tmp)):
        if(pairs[i, 0] == Y_tmp[j]):
          Y_tmp_train.append(0)
          X_tmp_train.append(X_tmp[j])
        elif(pairs[i, 1] == Y_tmp[j]):
          Y_tmp_train.append(1)
          X_tmp_train.append(X_tmp[j])

      Y_tmp_train = np.array(Y_tmp_train)
      X_tmp_train = np.array(X_tmp_train).T


      w, b = self.initialize_params(n)
      parameters, grads, costs = None, None, None  
      parameters, grads, costs = self.optimize(w, b, X_tmp_train, Y_tmp_train, 3000, 0.5, False)
      dict_params[frozenset(pairs[i])] = (parameters["w"], parameters["b"], costs)
      # print(parameters)
      # print(i)
      # print(pairs[i])
      # print('---')
    # print(dict_params)
    # Gradient descent (≈ 1 line of code)
    
    filename = 'params.dict'
    outfile = open(filename, 'wb')
    pickle.dump(dict_params, outfile)
    outfile.close()
    # Retrieve parameters w and b from dictionary "parameters"
    # w = parameters["w"]
    # b = parameters["b"]

    
    test = self.predict(self.X_test, dict_params, self.get_pairs())
    count = 0
    q = len(test)
    r = self.Y_test
    for i in range(q):
      if test[i] == r[0, i]:
        count += 1
    c = count/q
    print(c)


    # Predict test/train set examples (≈ 2 lines of code)
    # Y_prediction_test = predict(w, b, X_test)
    # Y_prediction_train = predict(w, b, X_train)

  def predict(self, X, params, class_pairs):
    """Retorna un vector de <(1,m)> con las etiquetas predecidas para las instancias en X 
    usando la técnica de reducción uno-contra-uno (one-vs-one).

    <class_pairs> contiene una lista con las k(k-1)/2 tuplas que identifican los 
    clasificadores que se utilizarán para la predicción. 

    <params> contiene un diccionario con los parámetros <w> y <b> de cada uno de los 
    clasificadores tal como se explica en la documentación del método <train_model>.

    A cada instancia en <X> se le predice una etiqueta por cada uno de los clasificadores 
    (modelos), es decir cada instancia tendrá inicialmente k(k-1)/2 etiquetas de las cuales 
    se escogerá la que más se repita como etiqueta definitiva para dicha instancia. Si se 
    repiten varias etiquetas un mismo número de veces, se escoge una al azar. Para hacer la 
    selección al azar use el módulo random de la librería numpy, y antes de hacer cada 
    llamado a una función de éste defina la semilla en 1 con <np.random.seed(1)>, esto es 
    para que obtenga resultados "aleatorios" iguales a los que espera el autograder.

    Se asume un umbral de 0.5 para la predicción.
    """
    predictions = []
    def local_predict(w_n,b_n,X_n):
      m = X_n.shape[1]
      Y_prediction = np.zeros((1,m))
      w_n = w_n.reshape(X_n.shape[0], 1)
    
      A = self.sigmoid(w_n.T.dot(X_n) + b_n)
    
      for i in range(A.shape[1]):
        Y_prediction[0,i] = 0 if (A[0,i] <= 0.5) else 1
    
      assert(Y_prediction.shape == (1, m))
      return Y_prediction

    for i in range (len(class_pairs)):
      params_local = params[frozenset(class_pairs[i])]
      w = params_local[0]
      b = params_local[1]
      
      prediction = local_predict(w, b, X)
      predictions.append(prediction)

    predictions = np.array(predictions)
    predictions = predictions.T

    arr = []
    np.random.seed(1)
    for i in range(predictions.shape[0]):
      nm = [0,0,0,0,0,0,0,0,0,0]
      for k in range(predictions.shape[2]):
        l = predictions[i, 0, k]
        tmp_l = int(l)
        nm[class_pairs[k][tmp_l]] += 1
      max_value = self.get_max(nm)
      if len(max_value) > 1:
        arr.append(max_value[np.random.randint(len(max_value))])
      else:
        arr.append(max_value[0])

    return arr

  def get_datasets(self):
    """Retorna un diccionario con los datasets preprocesados con los datos y 
    dimensiones que se usaron para el entrenamiento
    
    d = { "X_train": X_train,
    "X_test": X_test,
    "Y_train": Y_train,
    "Y_test": Y_test
    }
    """
    d = {
      "X_train": self.X_train,
      "X_test": self.X_test,
      "Y_train": self.Y_train,
      "Y_test": self.Y_test
    }
    return d

  def get_pairs(self):
    """Retorna una lista con las k(k-1)/2 tuplas que identifican los clasificadores que 
    conforman este modelo, ej.: [(0,1), (0,2) ... (8,9)]
    """
    return self.clasificadores

  def sigmoid(self, z):
    s = 1 / (1 + np.exp(-z))
    return s

  def initialize_params(self,dim):
    #np.random.seed(1)
    #w = np.random.rand(dim,1) * 0.01
    #b = np.random.randint(100) * 0.0001
    w = np.zeros((dim, 1))
    b = 0.0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

  def propagate(self, w, b, X, Y):
    m = X.shape[1]
    
    Z = w.T.dot(X) + b
    
    A = self.sigmoid(Z)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y)*np.log(1-A))
    
    dZ = A - Y
    dw = 1/m * X.dot(dZ.T)
    db = 1/m * np.sum(dZ)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return (grads, cost)
  
  def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):

      grads, cost = self.propagate(w, b, X, Y)

      dw = grads["dw"]
      db = grads["db"]
        
      w = w - learning_rate * dw
      b = b - learning_rate * db

      if i % 100 == 0:
        costs.append(cost)
        

      if print_cost and i % 100 == 0:
        print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

  
  def get_max(self, a):
    max_value = max(a)
    if a.count(max_value) > 1:
      return [i for i, x in enumerate(a) if x == max(a)]
    else:
      return [a.index(max(a))]
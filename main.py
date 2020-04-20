from hw_digit_recognizer import HWDigitRecognizer as clf
import pickle

#digitRecognizer = clf('autograder_data/mnist_train_0.01sampled.csv', 'autograder_data/mnist_test_0.01sampled.csv')
digitRecognizer = clf('datasets/mnist_train.csv', 'datasets/mnist_test.csv')
digitRecognizer.train_model()
print('end')
#data = digitRecognizer.get_datasets()
#file = open("params.dict", "rb")
#params = pickle.load(file)

#prediction = digitRecognizer.predict(data["X_test"], params, digitRecognizer.get_pairs())
#real = data["Y_test"]
#count = 0
#for i in range(len(real[0])):
#    if real[0, i] == prediction[i]:
#        count += 1
#print(count/100)

#   def get_max(self, a):
#     max_value = max(a)
#     if a.count(max_value) > 1:
#       return [i for i, x in enumerate(a) if x == max(a)]
#     else:
#       return [a.index(max(a))]


# for i in range(predictions.shape[0]):
#       nm = [0,0,0,0,0,0,0,0,0,0]
#       for k in range(predictions.shape[1]):
#         l = predictions[i, k]
#         nm[class_pairs[k, l]] += 1
#       max_value = self.get_max(nm)
#       if len(max_value) > 1:
#         arr.append(max_value[np.random.randint(len(max_value))])
#       else:
#         arr.append(max_value[0])
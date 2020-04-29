from hw_digit_recognizer import HWDigitRecognizer as clf
import pickle

digitRecognizer = clf('autograder_data/mnist_train_0.01sampled.csv', 'autograder_data/mnist_test_0.01sampled.csv')
# digitRecognizer = clf('datasets/mnist_train.csv', 'datasets/mnist_test.csv')
# digitRecognizer.train_model()
print('end')
data = digitRecognizer.get_datasets()
file = open("params.dict", "rb")
params = pickle.load(file)

prediction = digitRecognizer.predict(data["X_test"], params, digitRecognizer.get_pairs())
real = data["Y_test"]
count = 0
for i in range(len(real[0])):
   if real[0, i] == prediction[i]:
       count += 1
print(count/100)

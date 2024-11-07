import _pickle as cPickle
import gzip
import zipfile
import numpy as np
def load_data():
    with zipfile.ZipFile('file.zip', 'r') as zip_ref:
        zip_ref.extractall('D://DangTranTanLuc//Python//HandwritingRecognition//data//')
    print("abc")
    with open('D://DangTranTanLuc//Python//HandwritingRecognition//data//dataset.pkl', 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")

    print((training_data, validation_data, test_data))
    print("hello world")
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
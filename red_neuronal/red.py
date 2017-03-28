from load_data import *
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def clasificar(filename):
    cs = clases()
    x_data = normalizar(filename, 28, 28)
    #x_data = normalizar("../imagenes/staff_segments/level0/3.png", 28, 28)
    r = clf.predict(x_data.reshape(1, -1))
    print cs[r[0]]

train_data, train_labels = load_training_data()
test_data, test_labels = load_testing_data()
val_data, val_labels = load_validation_data()

train_data = train_data+val_data
train_labels = train_labels+val_labels

clf = MLPClassifier(solver='lbfgs', alpha=0.01,
                    hidden_layer_sizes=(300), random_state=1)
clf.fit(train_data, train_labels)
pred = clf.predict(test_data)
acc = accuracy_score(test_labels, pred)
print acc


for filename in os.listdir('../input'):
    if filename.endswith(".png"):
        clasificar('../input/'+filename)

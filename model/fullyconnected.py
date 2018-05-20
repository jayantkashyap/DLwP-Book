from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
# from datasets.dataset import dataset_loader, list_files


# PATH = '../datasets'
#
# data_files = list(list_files(PATH))
# data, labels = dataset_loader(data_files, verbose=500, preprocessing=(28, 28))
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# data = data.reshape(data.shape[0], 28*28)
# data = data.astype('float32')/255
#
# le = LabelEncoder()
# labels = le.fit_transform(labels)
# labels = to_categorical(labels)

# train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=101)

train_X = train_X.reshape((train_X.shape[0], 28*28))
train_X = train_X.astype('float32')/255

test_X = test_X.reshape((test_X.shape[0], 28*28))
test_X = test_X.astype('float32')/255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28*28, )))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_X, test_y)

print(f'test_acc: {test_acc}')
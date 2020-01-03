from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt


label = open("/content/drive/My Drive/Colabb/tek_etiket_karistirilmis.txt","r")
data = open("/content/drive/My Drive/Colabb/tek_veri_karistirilmis.txt","r")

def getDataSet(label, data, data_rate):
  data_array= []
  label_array = []

  #label için ayrıştırma
  for i in label:
    label_array = i.split(",")
  for k in range(len(label_array)):
    label_array[k] = int(label_array[k])

  #data için ayrıştırma
  for j in data:
    s = j.split(",")
    for i in range(len(s)):
      s[i] = int(s[i])
    data_array.append(np.array(s))
 
  length = int(len(data_array)*(data_rate/100))

  test_data = np.array(data_array[length:])
  test_label = np.array(label_array[length:])
  print(test_data.shape  )
  
  train_data = np.array(data_array[:length])
  train_label = np.array(label_array[:length])

  print(train_data)

  return train_data, train_label, test_data, test_label

train_data, train_labels, test_data, test_labels = getDataSet(label, data, 70)# %80 eğitim için ayrıldı %20 test için

def vectorize(sequences, dimension=10000):# verileri modelin anlayacağı hale getirir
  results = np.zeros((len(sequences),dimension))
  for i, sequence in enumerate(sequences):
    results[i,sequence] = 1.
  return results

# x-data y-label
x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# float 32 çünkü verilerin sonuç katmanında hataları hesaplanacak 0 ve 1 arası şeklinde

#modelimizi tanımlıyoruz
model = models.Sequential()
model.add(layers.Dense(1024, activation = 'relu', input_shape=(10000,)))
model.add(layers.Dense(2024, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
# tek çıkış katmanımız var çünkü çıktı katmanında 0 ve 1 arası bir değer çıkış yapacak

# doğrulama veri setini hazırlıyoruz
x_val = x_train[:5000]
partial_x_train = x_train[5000:]

y_val = y_train[:5000]
partial_y_train = y_train[5000:]

#modelimizi compile ediyoruz
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
# binary_crossentropy kayıp fonksiyonu hem ikili sınıflandırma hemde olasılık çiktısı olan modeller için en iyi seçimlerden biridir

#modelimizi eğitiyoruz
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=15,
                    batch_size=256,
                    validation_data=(x_val, y_val))


loss=history.history['loss']
val_loss=history.history['val_loss']
acc = history.history['acc']
val_acc =history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs, val_loss, 'bo', label = 'Egitim Kaybı')
plt.plot(epochs, loss, 'b', label='Dogruluk Kaybı')
plt.title('Egitim ve Dogruluk Kaybı')
plt.xlabel('Epoklar')
plt.ylabel('kayip')
plt.legend()

plt.savefig('/content/drive/My Drive/Colabb/dfg.png',dpi=400,edgecolor='b',)

plt.show()


import numpy as np
import keras

class LSTMDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, feature_vecs, labels, no_stations, no_cumul_samples, seq_length, batch_size=64, dim=(32,25088), n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.feature_vecs = feature_vecs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.no_stations = no_stations
        self.no_cumul_samples = no_cumul_samples
        self.seq_length = seq_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.feature_vecs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = np.empty((self.batch_size, *self.dim),dtype=float)
        y = np.empty((self.batch_size), dtype=int)
        
        for i, index in enumerate(batch_indices):
            for station in range(0,self.no_stations) :
                if ((index >= self.no_cumul_samples[station]) and (index < self.no_cumul_samples[station+1])) :
                    for row in range(index-self.seq_length+1, index+1) :
                        if (row<self.no_cumul_samples[station])  :
                            continue;
                        a=  self.feature_vecs[row].T
                        a=np.ravel(a)
                        X[i][row-index+self.seq_length-1]=a.T #fill up batch element input
                    y[i]=self.labels[index]
                    break

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.feature_vecs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

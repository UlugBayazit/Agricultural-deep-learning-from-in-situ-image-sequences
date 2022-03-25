
import numpy as np
import tensorflow.keras
from tensorflow.keras.preprocessing import image
from skimage.transform import resize

class CustomImageDataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, field, df, batch_size=32, dim=(1024,1024), n_channels=1,
                  shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = df[field]
        self.n_channels = n_channels
 #       self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.paths=df['path']

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
       
        img  = image.load_img(self.paths[indexes[0]])
        self.dim=(img.size[1],img.size[0])
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            img  = image.load_img(self.paths[ID])
            X[i,] = image.img_to_array(img)
#            if ((x.shape[0]!=max_x)or(x.shape[1]!=max_y)) :
#                X[i,] = resize(x, (max_x,max_y,3), order=3, anti_aliasing=True)
#            else :
#                X[i,] = x
            # Store class
            y[i] = self.labels[ID]

 #       return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X,y


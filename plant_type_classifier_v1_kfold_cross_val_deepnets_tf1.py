import array  
import numpy as np
import os
import gc
import sys
import tensorflow as tf
from batch_generator import LSTMDataGenerator
from scipy import io as IO
from keras import backend as K 
from keras import callbacks
from keras.models import Model
from keras.optimizers import Adam,Adagrad
from keras.layers import Input, Dropout, LSTM, Dense,TimeDistributed,GRU
from keras.utils import plot_model

class save_predictions(callbacks.Callback):
        def __init__(self,batch_size=None,x_test=None,z_test=None) :
            self.x_test=x_test
            self.z_test=z_test
            self.batch_size=batch_size
            
        def on_train_begin(self, logs={}) :
            self.predictions = []
            
        def on_epoch_end(self, epoch, logs={}):
            self.predictions.append(self.model.predict(self.x_test,self.batch_size))
            
def set_train_test_stations(fold_no,test_station_codes,test_station_types,train_station_codes,train_station_types,station_codes,station_types):
    
    for index in range(0,len(train_stations[fold_no])) :
        if (train_stations[fold_no][index]!=-1) :
            train_station_codes.append(station_codes[train_stations[fold_no][index]])
            train_station_types.append(station_types[train_stations[fold_no][index]])
        
    for index in range(0,len(test_stations[fold_no])) :
        if (test_stations[fold_no][index]!=-1) :
            test_station_codes.append(station_codes[test_stations[fold_no][index]])
            test_station_types.append(station_types[test_stations[fold_no][index]]) 

def process_fold_data2(fold_no,initial_weights2) :
    
    global graph
    with graph.as_default():
        adam_opt=Adam(lr=0.000025,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model2.compile(loss='categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy'])
        
        train_station_types=[]
        train_station_codes=[]
        test_station_types=[]
        test_station_codes=[]
    
        set_train_test_stations(fold_no,test_station_codes,test_station_types,train_station_codes,train_station_types,station_codes,station_types)
       
        no_test_stations=len(test_station_codes)
        no_train_stations=len(train_station_codes)
        
        no_samples=0
        
        trainmats = []
        testmats=[]
        no_cumul_samples=array.array('I',range(0,no_train_stations+1))
        no_cumul_samples[0]=no_samples
        if (is_fine_tune) :
            if (feature_epoch_no==-1):
                fine_tune="_fine_tune_type/fold_"+str(fold_no)+"_initial"+"/"
            else :
                fine_tune="_fine_tune_type/fold_"+str(fold_no)+"_"+str(feature_epoch_no)+"/"
        else :
            fine_tune="/"   
        for station in range(0,no_train_stations) :
    
            no_files=0
            if (network_type=="InceptionResnetV2") :  
                traindataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                traindataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                traindataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                traindataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                 traindataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                 traindataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
                 
            dir_files=os.listdir( traindataDir )
            dir_files.sort() #need to sort file names in training directory
            for file in dir_files :
                no_files+=1
                if (load_data_type=="MATLAB")   :
                    trainmats.append( IO.loadmat( traindataDir+file ) )
                elif (load_data_type=="python")   :
                    trainmats.append( np.load( traindataDir+file ) )
            no_samples+=no_files
            no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
            
    
        if (load_data_type=="MATLAB")   :
            a=trainmats[0]['spDesc']
        elif (load_data_type=="python")   :   
            a=trainmats[0].T
        
        z_train=np.ndarray(shape=(no_samples),dtype=int)
        z_train.fill(0)
        no_samples=0
        for station in range(0,no_train_stations) :
            for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                z_train[no_samples]=plant_types.index(train_station_types[station])
                no_samples+=1 
        
# Batch generator supplies the LSTM network with batches of windows of training feature vectors in memory      
        params = {'no_stations' : no_train_stations, 'no_cumul_samples' : no_cumul_samples, 'seq_length' : seq_length, 'batch_size': 64,  'dim' : (seq_length,data_dim), 'n_classes' : no_plant_types,'shuffle': True}
        training_generator = LSTMDataGenerator(trainmats, z_train, **params)
        
        no_samples=0   
        no_cumul_samples[0]=no_samples
        for station in range(0,no_test_stations) :
            no_files=0  
            if (network_type=="InceptionResnetV2") :  
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                 testdataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                 testdataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
    
            dir_files=os.listdir( testdataDir )
            dir_files.sort() #need to sort file names in training directory
            for file in dir_files :
                no_files+=1
                if (load_data_type=="MATLAB")   :
                    testmats.append( IO.loadmat( testdataDir+file ) )
                elif (load_data_type=="python")   :
                    testmats.append( np.load( testdataDir+file ) )
            no_samples+=no_files
            no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
           
        if (load_data_type=="MATLAB")   :
            a=testmats[0]['spDesc']
        elif (load_data_type=="python")   :   
            a=testmats[0].T
    
        x_test=np.ndarray(shape=(no_samples,seq_length,data_dim),dtype=float)
        z_test=np.ndarray(shape=(no_samples),dtype=int)
        z_test.fill(0)
        
        no_samples=0
        for station in range(0,no_test_stations) :
            for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                for row in range(index-seq_length+1, index+1) :
                    if (row<no_cumul_samples[station])  :
                        continue;
                    if (load_data_type=="MATLAB")   :
                        a=testmats[row]['spDesc']
                    elif (load_data_type=="python")   :   
                        a=testmats[row].T
                        a=np.ravel(a)
                    x_test[no_samples][row-index+seq_length-1]=a.T    
                z_test[no_samples]=plant_types.index(test_station_types[station])  
                no_samples+=1

        if (is_type_output):
            predictions_log=save_predictions(no_samples,x_test,z_test)
        if (fold_no!=first_fold) :
            model2.set_weights(initial_weights2) 

        
# Batch generator supplies the LSTM network with batches of windows of test feature vectors in memory         
        params = {'no_stations' : no_test_stations, 'no_cumul_samples' : no_cumul_samples, 'seq_length' : seq_length, 'batch_size': 64,  'dim' : (seq_length,data_dim), 'n_classes' : no_plant_types,'shuffle': True}
        validation_generator = LSTMDataGenerator(testmats, z_test, **params)
        if (is_type_output):
            history2=model2.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=no_epochs2,
                        callbacks=[predictions_log])
        else :
            history2=model2.fit_generator(generator=training_generator,
                        validation_data=validation_generator, epochs=no_epochs2)
            
        if (is_save_weights_folds)  :
                if (is_selu) :
                    model2.save_weights('plant_classifier_model_weights_feature_epoch_no'+str1+append_str+'_lstm_selu'+'_fold'+str(fold_no)+network_type+'.h5')
                else :
                    model2.save_weights('plant_classifier_model_weights_feature_epoch_no'+str1+append_str+'_fold'+str(fold_no)+network_type+'.h5')
                
        z_predict=model2.predict(x_test)
        acc2=0
        for sample in range(0,no_samples) :
            if (z_test[sample]==np.argmax(z_predict[sample])) : 
                acc2+=1

        acc2=acc2/no_samples
        print("acc2 %.3f%%" % (acc2*100))  
        if (is_type_output):    
            return predictions_log.predictions,history2.history['val_acc'],history2.history['val_loss'],history2.history['acc'],history2.history['loss']
        else :
            return history2.history['val_acc'],history2.history['val_loss'],history2.history['acc'],history2.history['loss']
        K.clear_session() 
        gc.collect()

_defined_contrast_stretch=False
if (_defined_contrast_stretch==True)  :
    append_str='contrast_stretch'
else :
    append_str=''
        
is_selu=False
is_output_history = True
is_save_model=True
is_save_weights_folds=True
feature_epoch_no=3 #Read the set of feature vectors fine tuned for feature_epoch_no epochs

is_type_output= False

if (is_type_output): 
    total_predictions=[]

is_fine_tune=True
network_type="DenseNet201"
if (network_type=="VGG16") :
    load_data_type="python"
    data_dim=25088
elif (network_type=="VGG19") :
    load_data_type="python"
    data_dim=4096
elif (network_type=="InceptionResnetV2") :
    load_data_type="python"
    data_dim=1536
elif (network_type=="ResNet101") :
    load_data_type="python"
    data_dim=100352
elif (network_type=="EfficientNetB7") :
    load_data_type="python"
    data_dim=125440
elif (network_type=="DenseNet201") :
    load_data_type="python"
    data_dim=94080

fold_no_start=int(sys.argv[1])    
fold_no_end=int(sys.argv[2]) 

normalize_stages=0
seq_length=32
no_epochs2=25

day_of_year=[0,31,59,90,120,151,181,212,243,273,304,334]
plant_types=["Arpa","Aycicegi","Bugday","Misir","Nohut","Pamuk"]
no_plant_types=len(plant_types)
no_stages={"Arpa":9,"Bugday":9,"Aycicegi":7,"Misir":8,"Nohut":7,"Pamuk":8}

station_types = [line.rstrip() for line in open('station_types.txt')]
station_codes = [line.rstrip() for line in open('station_codes.txt')]     

common_input=Input(shape=(seq_length, data_dim))
slice_input=Input(shape=(data_dim,))
slice_ann_hidden=Dropout(0.2)(Dense(32,kernel_initializer='glorot_uniform',activation="tanh")(Dropout(0.2)(slice_input)))
#slice_ann_output=Dropout(0.2)(Dense(4,kernel_initializer='glorot_uniform',activation="tanh")(slice_ann_hidden))
model_slice_ann=Model(inputs=slice_input,outputs=slice_ann_hidden)
seq_ann_output=TimeDistributed(model_slice_ann)(common_input)

if (is_selu) :
    seq_ann_1D_output1=Dropout(0.2)(LSTM(25,input_shape=(None, 32),dropout=0.2,recurrent_dropout=0.2, unit_forget_bias=True,activation="selu")(seq_ann_output))
else:
    seq_ann_1D_output1=Dropout(0.2)(LSTM(25,input_shape=(None, 32),dropout=0.2,recurrent_dropout=0.2, unit_forget_bias=True)(seq_ann_output))

plant_type_output=Dense(no_plant_types, kernel_initializer='glorot_uniform',activation="softmax")(seq_ann_1D_output1)
model2=Model(inputs=common_input,outputs=plant_type_output)

model_name='plant_classifier_model'
if (is_selu) :
    model_name=model_name+'_lstm_selu' 
model_name=model_name+'.json'   
if (is_save_model) :
    if (os.path.exists(model_name)==False) :
        model_json = model2.to_json()
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
            
#plot_model(model_slice_ann,to_file='model_slice_ann.png',show_shapes=True)
#plot_model(model2,to_file='model2.png',show_shapes=True)
            
train_stations=np.loadtxt("train-partition",dtype='int')
test_stations=np.loadtxt("test-partition",dtype='int')
no_folds=train_stations.shape[0]
total_acc=[]

first_fold=fold_no_start #temporary for code testing purpose
initial_weights2=model2.get_weights()
if (is_fine_tune) :
    if (feature_epoch_no==-1):
        str1=str(seq_length)+"_initial"
    else :
        str1=str(seq_length)+"_"+str(feature_epoch_no)

graph = tf.get_default_graph()
for fold_no in range(fold_no_start,fold_no_end) :
    if (is_output_history) :
        if (is_selu) :
            output_history2=open('output_history2_winlength_32'+str1+append_str+'_lstm_selu'+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+network_type+'.txt',mode='a')
        else :
            output_history2=open('output_history2_winlength_32'+str1+append_str+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+network_type+'.txt',mode='a')
        output_history2.write("\n")
    if (is_type_output):
        predictions_fold,acc_fold,loss_fold,train_acc_fold,train_loss_fold=process_fold_data2(fold_no,initial_weights2) 
        total_predictions.append(predictions_fold)
    else :
        acc_fold,loss_fold,train_acc_fold,train_loss_fold=process_fold_data2(fold_no,initial_weights2) 
   
    total_acc.append(acc_fold)
    if (is_output_history) :
        for ep in range(0,no_epochs2) :
            output_history2.write(str(acc_fold[ep]))
            output_history2.write(" ")
        output_history2.write("\n")
        for ep in range(0,no_epochs2) :
            output_history2.write(str(loss_fold[ep]))
            output_history2.write(" ")
        output_history2.write("\n")
        for ep in range(0,no_epochs2) :
            output_history2.write(str(train_acc_fold[ep]))
            output_history2.write(" ")
        output_history2.write("\n")
        for ep in range(0,no_epochs2) :
            output_history2.write(str(train_loss_fold[ep]))
            output_history2.write(" ")
        output_history2.close()

if (is_type_output):
    max_ep=0
    for ep in range(0,no_epochs2) :
        sum_ep=0
     
        count_processed_folds=0
        for fold_no in range(first_fold,no_folds) :
    
            sum_ep+=total_acc[count_processed_folds][ep]
            count_processed_folds+=1
            
        sum_ep/=no_folds 
        if (sum_ep>max_ep) :
            max_ep=sum_ep
            best_ep2=ep    
#print("best_ep2 %d" % best_ep2) 

    save_predictions_file=open('new_save_predictions_file'+str1+'.txt',mode='w')
    count_processed_folds=0
    for fold_no in range(first_fold,no_folds) :
        for i in range(0,int(total_predictions[count_processed_folds][best_ep2].size/total_predictions[count_processed_folds][best_ep2][0].size)) :
            for j in range(0,total_predictions[count_processed_folds][best_ep2][i].size) :
                string= "{:.9f}".format(total_predictions[count_processed_folds][best_ep2][i][j])
                save_predictions_file.write(string + ',')
    
        count_processed_folds+=1
    save_predictions_file.close()


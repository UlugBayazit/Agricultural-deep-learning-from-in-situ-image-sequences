import array  
import numpy as np
import scipy
import os
import sys
import tensorflow as tf
import gc
from batch_generator_phenology import LSTMPhenologyDataGenerator,LSTMPhenologyJointDataGenerator
from keras import backend as K 
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, LSTM, Dense,concatenate
from keras.utils import to_categorical
#from keras.utils import plot_model
    
def set_train_test_stations(fold_no,test_station_codes,test_station_types,train_station_codes,train_station_types,station_codes,station_types):
    
    for index in range(0,len(train_stations[fold_no])) :
        if (train_stations[fold_no][index]!=-1) :
            train_station_codes.append(station_codes[train_stations[fold_no][index]])
            train_station_types.append(station_types[train_stations[fold_no][index]])
        
    for index in range(0,len(test_stations[fold_no])) :
        if (test_stations[fold_no][index]!=-1) :
            test_station_codes.append(station_codes[test_stations[fold_no][index]])
            test_station_types.append(station_types[test_stations[fold_no][index]]) 

def process_fold_data1(fold_no,initial_weights1) :
   
    global graph
    with graph.as_default():
       
        adam_opt=Adam(lr=0.000025,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
        model1.compile(loss='mean_absolute_error',optimizer=adam_opt)
        
        day_stage={}
        
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
        if (is_fine_tune) :
            if (feature_epoch_no==-1):
                fine_tune="_fine_tune_phenology/fold_"+str(fold_no)+"_initial"+"/"
            else :
                fine_tune="_fine_tune_phenology/fold_"+str(fold_no)+"_"+str(feature_epoch_no)+"/"
        else :
            fine_tune="/"
        no_cumul_samples=array.array('I',range(0,no_train_stations+1))
        no_cumul_samples[0]=no_samples
        for station in range(0,no_train_stations) :
            day_stage[station]=list(0 for x in range(0,no_stages[train_station_types[station]]))
            no_files=0
            t=train_station_codes[station].split("_")
            train_station_code_base=t[0]
            trainlabelfile="../../bap_proje/CNNFeat_VGG"+"/"+train_station_types[station]+"/"+train_station_types[station]+".txt"
           
            f = open(trainlabelfile, "r")
            for line in f:
                words = line.split()
                if words == [] :
                    continue
                if words[1]==train_station_code_base :
                    foundflag='True'
                    for stage in range(0,no_stages[train_station_types[station]])  :
                        date=f.readline().split(".")
                        day_stage[station][stage]=int(date[0])+day_of_year[int(date[1])-1] 
                    break
            f.close()  
            if foundflag=='False' :
                print("\n Station not found")
                exit
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
                    trainmats.append( scipy.io.loadmat( traindataDir+file ) )
                elif (load_data_type=="python")   :
                    trainmats.append( np.load( traindataDir+file ) )
            no_samples+=no_files
            no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
            
        for station in range(0,no_train_stations) :
            for stage in range(0,no_stages[train_station_types[station]]-1)  :
                if day_stage[station][stage]>day_stage[station][stage+1] :
                    for stage1 in range(stage+1,no_stages[train_station_types[station]])  :
                        day_stage[station][stage1]+=365
                    break
        if (load_data_type=="MATLAB")   :
            a=trainmats[0]['spDesc']
        elif (load_data_type=="python")   :   
            a=trainmats[0].T

        y_train=np.ndarray(shape=(no_samples,1),dtype=float)
        y_train.fill(0)
        if (is_plant_type_input) :
            z_train_predict=np.ndarray(shape=(no_samples,no_plant_types),dtype=float) 
            z_train=np.ndarray(shape=(no_samples),dtype=int)
        no_samples=0
        for station in range(0,no_train_stations) :
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
            for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                filename_decomposed=dir_files[index-no_cumul_samples[station]].split("-")
                file_day=day_of_year[int(filename_decomposed[1])-1]+int(filename_decomposed[2])
                if file_day<day_stage[station][0] : 
                    file_day+=365
                for stage in range(0,no_stages[train_station_types[station]]-1)  :
                    if (day_stage[station][stage]<=file_day)and(file_day<day_stage[station][stage+1]) :
                        y_train[no_samples]=stage+(file_day-day_stage[station][stage])/(day_stage[station][stage+1]-day_stage[station][stage])
                        if (normalize_stages)   :
                            y_train[no_samples]=y_train[no_samples]*max_no_stages/no_stages[train_station_types[station]]
                        break
                if (y_train[no_samples]==0) :
                    if (day_stage[station][no_stages[train_station_types[station]]-1]<=file_day) :
                       y_train[no_samples]=no_stages[train_station_types[station]]-1
                       if (normalize_stages)   :
                            y_train[no_samples]=y_train[no_samples]*max_no_stages/no_stages[train_station_types[station]]

                if (is_plant_type_input) :
                    z_train[no_samples]=plant_types.index(train_station_types[station])
                no_samples+=1 
        if (is_plant_type_input) :        
            if (is_predict_type) :   
                no_samples=0
                for station in range(0,no_train_stations) :
                    for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                        for j in range(0,no_plant_types) :
                            string=load_predictions_file.read(12)
                            z_train_predict[no_samples][j]=float(string[0:10])
                        no_samples+=1
                
            else :
                z_train_predict=to_categorical(z_train,no_plant_types)
                
# Batch generator supplies the LSTM network with batches of windows of training feature vectors in memory            
        params = {'no_stations' : no_train_stations, 'no_cumul_samples' : no_cumul_samples, 'seq_length' : seq_length, 'batch_size': 64,  'dim' : (seq_length,data_dim), 'n_classes' : no_plant_types,'shuffle': True}
        if (is_plant_type_input) :        
            training_generator = LSTMPhenologyJointDataGenerator(trainmats, z_train, y_train, **params)
        else :
            training_generator = LSTMPhenologyDataGenerator(trainmats, y_train, **params)    
            
        no_samples=0   
        no_cumul_samples[0]=no_samples
        for station in range(0,no_test_stations) :
            no_files=0
            t=test_station_codes[station].split("_")
            test_station_code_base=t[0]
            testlabelfile="../../bap_proje/CNNFeat_VGG"+"/"+test_station_types[station]+"/"+test_station_types[station]+".txt"
            
            f = open(testlabelfile, "r")
            for line in f:
                words = line.split()
                if words == [] :
                    continue
                if words[1]==test_station_code_base :
                    foundflag='True'
                    for stage in range(0,no_stages[test_station_types[station]])  :
                        date=f.readline().split(".")
                        day_stage[station][stage]=int(date[0])+day_of_year[int(date[1])-1] 
                    break
                
            f.close()  
            if foundflag=='False' :
                print("\n Station not found")
                exit
                
            if (network_type=="InceptionResnetV2") :  
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            
            dir_files=os.listdir( testdataDir )
            dir_files.sort() #need to sort file names in training directory
            for file in dir_files :
                no_files+=1
                if (load_data_type=="MATLAB")   :
                    testmats.append( scipy.io.loadmat( testdataDir+file ) )
                elif (load_data_type=="python")   :
                    testmats.append( np.load( testdataDir+file ) )
            no_samples+=no_files
            no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
        
        for station in range(0,no_test_stations) :
            for stage in range(0,no_stages[test_station_types[station]]-1)  :
                if day_stage[station][stage]>day_stage[station][stage+1] :
                    for stage1 in range(stage+1,no_stages[test_station_types[station]])  :
                        day_stage[station][stage1]+=365
                    break    
        if (load_data_type=="MATLAB")   :
            a=testmats[0]['spDesc']
        elif (load_data_type=="python")   :   
            a=testmats[0].T

        y_test=np.ndarray(shape=(no_samples,1),dtype=float)
        y_test.fill(0)
        if (is_plant_type_input) :
            z_test_predict=np.ndarray(shape=(no_samples,no_plant_types),dtype=float) 
            z_test=np.ndarray(shape=(no_samples),dtype=int)
            z_test.fill(0.001)
        no_samples=0
        for station in range(0,no_test_stations) :
            if (network_type=="InceptionResnetV2") :  
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                testdataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            
            dir_files=os.listdir( testdataDir )
            dir_files.sort() #need to sort file names in test directory
            for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                filename_decomposed=dir_files[index-no_cumul_samples[station]].split("-")
                file_day=day_of_year[int(filename_decomposed[1])-1]+int(filename_decomposed[2])
                if file_day<day_stage[station][0] : 
                    file_day+=365
                for stage in range(0,no_stages[test_station_types[station]]-1)  :
                    if (day_stage[station][stage]<=file_day)and(file_day<day_stage[station][stage+1]) :
                        y_test[no_samples]=stage+(file_day-day_stage[station][stage])/(day_stage[station][stage+1]-day_stage[station][stage])
                        if (normalize_stages)   :
                            y_test[no_samples]=y_test[no_samples]*max_no_stages/no_stages[test_station_types[station]]
                        break
                if (y_test[no_samples]==0) :
                    if (day_stage[station][no_stages[test_station_types[station]]-1]<=file_day) :
                       y_test[no_samples]=no_stages[test_station_types[station]]-1
                       if (normalize_stages)   :
                            y_test[no_samples]=y_test[no_samples]*max_no_stages/no_stages[test_station_types[station]]    

                if (is_plant_type_input) :
                    z_test[no_samples]=plant_types.index(test_station_types[station])  
                no_samples+=1
        print("\n Fold {0} size {1}".format(fold_no,no_samples))
        if (is_plant_type_input) :
            if (is_predict_type) :
                no_samples=0 
                for station in range(0,no_test_stations) :
                    for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
                        for j in range(0,no_plant_types) :
                            string=load_predictions_file.read(12)
                            z_test_predict[no_samples][j]=float(string[0:10])
                        no_samples+=1        
            else :
                z_test_predict=to_categorical(z_test,no_plant_types)
    
        if (fold_no!=first_fold) :
            model1.set_weights(initial_weights1) 
        
# Batch generator supplies the LSTM network with batches of windows of test feature vectors in memory              
        params = {'no_stations' : no_test_stations, 'no_cumul_samples' : no_cumul_samples, 'seq_length' : seq_length, 'batch_size': 64,  'dim' : (seq_length,data_dim), 'n_classes' : no_plant_types,'shuffle': True}
        if (is_plant_type_input) :
            validation_generator = LSTMPhenologyJointDataGenerator(testmats, z_test, y_test, **params)    
        else :
            validation_generator = LSTMPhenologyDataGenerator(testmats, y_test, **params) 
       
        if (is_plant_type_input) :
            
            history1=model1.fit_generator(generator=training_generator,
                    validation_data=validation_generator, epochs=no_epochs1,
            #                    use_multiprocessing=True,workers=6,
                                )
            #        history1=model1.fit([x_train,z_train_predict], y_train,           
            #          batch_size=no_samples, epochs=no_epochs1, verbose=1,           
            #          validation_data=([x_test,z_test_predict], y_test)) 
        else :
            
            history1=model1.fit_generator(generator=training_generator,
                    validation_data=validation_generator, epochs=no_epochs1,
            #                    use_multiprocessing=True,workers=6,
                        )
                
        #        history1=model1.fit(x_train, y_train,           
        #          batch_size=no_samples, epochs=no_epochs1, verbose=1,           
        #          validation_data=(x_test, y_test)) 
            
                
        hist_array1[fold_no]=history1.history['val_loss']
        
        if (is_save_weights_folds)  :
            if (is_selu) :
                model1.save_weights('phenology_classifier_model_weights_feature_epoch_no'+str1+append_str+'_lstm_selu'+'_fold'+str(fold_no)+'.h5')
            else :
                model1.save_weights('phenology_classifier_model_weights_feature_epoch_no'+str1+append_str+'_fold'+str(fold_no)+'.h5')
            
        return history1.history['val_loss'],history1.history['loss']
        K.clear_session()
        gc.collect()
        
_defined_contrast_stretch=True
if (_defined_contrast_stretch==True)  :
    append_str='contrast_stretch'
else :
    append_str=''
    
is_selu=True
is_output_history = True
is_save_model=True
is_save_weights_folds=True
feature_epoch_no=5 #Read the set of feature vectors fine tuned for feature_epoch_no epochs

is_plant_type_input=False   
is_predict_type=False

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

normalize_stages=1
seq_length=32
no_epochs1=100 

day_of_year=[0,31,59,90,120,151,181,212,243,273,304,334]
plant_types=["Arpa","Aycicegi","Bugday","Misir","Nohut","Pamuk"]
no_plant_types=len(plant_types)
no_stages={"Arpa":9,"Bugday":9,"Aycicegi":7,"Misir":8,"Nohut":7,"Pamuk":8}

max_no_stages=0
for plant_type in range(0, no_plant_types) :
    if (no_stages[plant_types[plant_type]]>max_no_stages)  :
        max_no_stages=no_stages[plant_types[plant_type]]


station_types = [line.rstrip() for line in open('station_types.txt')]
station_codes = [line.rstrip() for line in open('station_codes.txt')]    

common_input=Input(shape=(seq_length, data_dim))

if (is_selu) :
    lstm_output=LSTM (25,input_shape=(None, data_dim),dropout=0.2,recurrent_dropout=0.2, unit_forget_bias=True,activation="selu")(common_input)
else :
    lstm_output=LSTM (25,input_shape=(None, data_dim),dropout=0.2,recurrent_dropout=0.2, unit_forget_bias=True)(common_input)

if (is_plant_type_input) :    #one extra layer used if plant_type_input is fed 
    plant_type_input=Input(shape=(no_plant_types,))  #separately feed in generated and collected plant_type_output as plant_type_input
    concatenated_phase_vars=concatenate([plant_type_input,lstm_output])
    phase_hidden=Dropout(0.2)(Dense(4,kernel_initializer='glorot_uniform',activation="relu")(concatenated_phase_vars))
    phase_output=Dense(1, kernel_initializer='glorot_uniform',activation="linear")(phase_hidden)
    model1=Model(inputs=[common_input,plant_type_input], outputs=phase_output)
else :    
    phase_output=Dense(1, kernel_initializer='glorot_uniform',activation="linear")(lstm_output)
    model1=Model(inputs=common_input, outputs=phase_output)

model_name='phenology_classifier_model'
if (is_selu) :
    model_name=model_name+'_lstm_selu' 
model_name=model_name+'.json'   
if (is_save_model) :
    if (os.path.exists(model_name)==False) :
        model_json = model1.to_json()
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
        
#plot_model(model_slice_ann,to_file='model_slice_ann.png',show_shapes=True)
#plot_model(model1,to_file='model1.png',show_shapes=True)

initial_weights1=model1.get_weights()

train_stations=np.loadtxt("train-partition",dtype='int')
test_stations=np.loadtxt("test-partition",dtype='int')
no_folds=train_stations.shape[0]
hist_array1=np.ndarray(shape=(no_folds,no_epochs1),dtype=float)

first_fold=fold_no_start 
if (is_predict_type) :
    load_predictions_file=open('save_predictions_file.txt',mode='r')
if (is_fine_tune) :
    if (feature_epoch_no==-1):
        str1=str(seq_length)+"_initial"
    else :
        str1=str(seq_length)+"_"+str(feature_epoch_no)    

graph = tf.get_default_graph()
for fold_no in range(fold_no_start,fold_no_end) :
    if (is_output_history) :
        if (is_selu) :
            output_history1=open('output_history1_winlength_32_corrected'+str1+append_str+'_lstm_selu'+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+'.txt',mode='a')
        else :
            output_history1=open('output_history1_winlength_32_corrected'+str1+append_str+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+'.txt',mode='a')
        output_history1.write("\n")
    loss_fold,train_loss_fold=process_fold_data1(fold_no,initial_weights1) 
    if (is_output_history) :
        for ep in range(0,no_epochs1) :
            output_history1.write(str(loss_fold[ep]))
            output_history1.write(" ")
        output_history1.write("\n")
        for ep in range(0,no_epochs1) :
            output_history1.write(str(train_loss_fold[ep]))
            output_history1.write(" ")
        output_history1.close()
    
if (is_predict_type) :        
    load_predictions_file.close() 

if (is_output_history) :
    if (is_selu) :
        output_history1=open('output_history1_winlength_32_corrected'+str1+append_str+'_lstm_selu'+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+'.txt',mode='a')
    else :
        output_history1=open('output_history1_winlength_32_corrected'+str1+append_str+'folds_'+str(fold_no_start)+'_'+str(fold_no_end)+'.txt',mode='a')
min_ep=1E32
for ep in range(0,no_epochs1) :
    sum_ep=0
    
    for fold_no in range(fold_no_start,fold_no_end) :
        sum_ep+=hist_array1[fold_no][ep]
    sum_ep/=no_folds 
    if (sum_ep<min_ep) :
        min_ep=sum_ep
        best_ep1=ep

print("best_ep1 %d " % best_ep1)
if (is_output_history) :
    output_history1.write(str(best_ep1))    
    output_history1.write(str(min_ep)) 
    output_history1.close()



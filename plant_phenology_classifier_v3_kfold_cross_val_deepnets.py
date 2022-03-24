import sys
import array  
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Dropout
#from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image

from skimage import exposure
from skimage.transform import resize
from skimage.color import rgb2yuv,yuv2rgb
from CustomizedImageDataGenerator_tf2 import CustomImageDataGenerator



def contrast_stretching(img):

    img_contrast_stretch = exposure.rescale_intensity(img)
    return img_contrast_stretch
def luminance_stretching(img):
    yuv_img=rgb2yuv(img)
    yuv_contrast_stretch=np.ndarray(shape=yuv_img.shape,dtype=float, order='C')
    yuv_contrast_stretch[1] = np.copy(yuv_img[1])
    yuv_contrast_stretch[2] = np.copy(yuv_img[2])
    yuv_contrast_stretch[0] = exposure.rescale_intensity(yuv_img[0])

    img_contrast_stretch=yuv2rgb(yuv_contrast_stretch)
    return img_contrast_stretch   
            
def set_train_test_stations(fold_no,test_station_codes,test_station_types,train_station_codes,train_station_types,station_codes,station_types):
    
    for index in range(0,len(train_stations[fold_no])) :
        if (train_stations[fold_no][index]!=-1) :
            train_station_codes.append(station_codes[train_stations[fold_no][index]])
            train_station_types.append(station_types[train_stations[fold_no][index]])
        
    for index in range(0,len(test_stations[fold_no])) :
        if (test_stations[fold_no][index]!=-1) :
            test_station_codes.append(station_codes[test_stations[fold_no][index]])
            test_station_types.append(station_types[test_stations[fold_no][index]])            

def process_fold_data2(fold_no,pass_no) :
   
  
    adam_opt=Adam(lr=0.000025,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
 
    if (network_type=="Resnet") :    
        base_model = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))    
    elif (network_type=="VGG16") :
        base_model = VGG16(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))
    elif (network_type=="VGG19") :
        base_model = VGG19(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))
    elif (network_type=="ResNet101") :
        base_model = ResNet101(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))
    elif (network_type=="EfficientNetB7") :
        base_model = EfficientNetB7(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))
    elif (network_type=="DenseNet201") :
        base_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(target_height,target_width,3))
   
    truncated_model=base_model
    #base_model.summary()
    #plot_model(truncated_model,to_file='truncated_model.png',show_shapes=True)
    inp=tf.keras.layers.Input(shape=(target_height,target_width,3))
    truncated_out=truncated_model(inp)
    slice_ann_1D_output2 = Dropout(0.2)(Dense(24, kernel_initializer='glorot_uniform',activation="relu")(Dropout(0.2)(Flatten()(truncated_out))))
    plant_phase_output=Dense(1, kernel_initializer='glorot_uniform',activation="linear")(slice_ann_1D_output2)
    full_model=Model(inp, plant_phase_output) 
    #full_model.summary()
    #plot_model(full_model,to_file='full_model.png',show_shapes=True)
    
    full_model.compile(loss='mean_absolute_error',optimizer=adam_opt,metrics=['mean_absolute_error'])
    
    day_stage={}
    train_station_types=[]
    train_station_codes=[]
    test_station_types=[]
    test_station_codes=[]

    set_train_test_stations(fold_no,test_station_codes,test_station_types,train_station_codes,train_station_types,station_codes,station_types)

    no_test_stations=len(test_station_codes)
    no_train_stations=len(train_station_codes)

    train_dir_files=[]
    max_no_train_samples=0
    train_image_path_list=[]
    image_station_list=[]
    for station in range(0,no_train_stations):
        
        traindataDir = "../../keras_cropped_data/"+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
        station_dir_files=os.listdir( traindataDir )
        station_dir_files.sort() #need to sort file names in training directory
        train_dir_files.append(station_dir_files)
        
        for file in train_dir_files[station] :
            if (file=='Thumbs.db') :
                continue
            image_station_list.append(station)
            train_image_path_list.append(traindataDir+file)
            max_no_train_samples+=1

    no_cumul_samples=array.array('I',range(0,no_train_stations+1))
    no_samples=0
    no_cumul_samples[0]=no_samples
    for station in range(0,no_train_stations) :
        day_stage[station]=list(0 for x in range(0,no_stages[train_station_types[station]]))
        no_files=0
        t=train_station_codes[station].split("_")
        train_station_code_base=t[0]
          
        traindataDir = "../../keras_cropped_data/"+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
        trainlabelfile="../../bap_proje/CNNFeat_VGG/"+train_station_types[station]+"/"+train_station_types[station]+".txt"
       
        foundflag='False'
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
        for file in train_dir_files[station] :
            if (file=='Thumbs.db') :
                continue
            no_files+=1
        no_samples+=no_files
        no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
    for station in range(0,no_train_stations) :
        for stage in range(0,no_stages[train_station_types[station]]-1)  :
            if day_stage[station][stage]>day_stage[station][stage+1] :
                for stage1 in range(stage+1,no_stages[train_station_types[station]])  :
                    day_stage[station][stage1]+=365
                break        
       
    y_train=np.ndarray(shape=(no_samples),dtype=float)
    y_train.fill(0)
    no_samples=0
    for station in range(0,no_train_stations) :
        traindataDir = "../../keras_cropped_data/"+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
        dir_files=os.listdir( traindataDir )
        dir_files.sort() #need to sort file names in training directory
        this_sample=0
        for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
            while (dir_files[this_sample]=='Thumbs.db') :
                this_sample+=1
            filename_decomposed=dir_files[this_sample].split("-")
            file_day=day_of_year[int(filename_decomposed[1])-1]+int(filename_decomposed[2])
            if file_day<day_stage[station][0] : #assume no plant cultivation takes more than a year, any day before first stage day belongs to next year
                file_day+=365
            for stage in range(0,no_stages[train_station_types[station]]-1)  :
                if (day_stage[station][stage]<=file_day)and(file_day<day_stage[station][stage+1]) :
                    y_train[no_samples]=stage+(file_day-day_stage[station][stage])/(day_stage[station][stage+1]-day_stage[station][stage])
                    if (normalize_stages)   :
                        y_train[no_samples]=y_train[no_samples]*max_no_stages/(no_stages[train_station_types[station]]-1)
                    break
            if (y_train[no_samples]==0) :
                if (day_stage[station][no_stages[train_station_types[station]]-1]<=file_day) :
                   y_train[no_samples]=no_stages[train_station_types[station]]-1
                   if (normalize_stages)   :
                        y_train[no_samples]=y_train[no_samples]*max_no_stages/(no_stages[train_station_types[station]]-1)
            no_samples+=1
            this_sample+=1
# Send batches to network directly from disk via paths stored in pandas dataframe            
    df_train = pd.DataFrame({'path': train_image_path_list,  'plant_phenology_stage': y_train})
#    train_generator=train_datagen.flow_from_dataframe(dataframe=df_train,  x_col='path', y_col='plant_phenology_stage', class_mode='other', target_size=(target_height,target_width), batch_size=batch_size,interpolation='lanczos')
    field='plant_phenology_stage'
    train_generator=CustomImageDataGenerator(field,df=df_train,batch_size=batch_size,dim=(2048,2048),n_channels=3)   
    test_dir_files=[]
    max_no_test_samples=0
    test_image_path_list=[]
    for station in range(0,no_test_stations) :
        
        testdataDir = "../../keras_cropped_data/"+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
        station_dir_files=os.listdir( testdataDir )
        station_dir_files.sort() #need to sort file names in training directory
        test_dir_files.append(station_dir_files)
        
        for file in test_dir_files[station] :
            if (file=='Thumbs.db') :
                continue
            image_station_list.append(station)
            test_image_path_list.append(testdataDir+file)
            max_no_test_samples+=1

    no_samples=0
    no_cumul_samples[0]=no_samples
    for station in range(0,no_test_stations) :
        day_stage[station]=list(0 for x in range(0,no_stages[test_station_types[station]]))
        no_files=0
        t=test_station_codes[station].split("_")
        test_station_code_base=t[0]
          
        testdataDir = "../../keras_cropped_data/"+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
        testlabelfile="../../bap_proje/CNNFeat_VGG/"+test_station_types[station]+"/"+test_station_types[station]+".txt"
       
        foundflag='False'
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
        for file in test_dir_files[station] :
            if (file=='Thumbs.db') :
                continue
            no_files+=1
        no_samples+=no_files
        no_cumul_samples[station+1]=no_cumul_samples[station]+no_files
    for station in range(0,no_test_stations) :
        for stage in range(0,no_stages[test_station_types[station]]-1)  :
            if day_stage[station][stage]>day_stage[station][stage+1] :
                for stage1 in range(stage+1,no_stages[test_station_types[station]])  :
                    day_stage[station][stage1]+=365
                break        
    y_test=np.ndarray(shape=(no_samples),dtype=float)
    y_test.fill(0)
    no_samples=0   
    for station in range(0,no_test_stations) :
        testdataDir = "../../keras_cropped_data/"+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
        dir_files=os.listdir( testdataDir )
        dir_files.sort() #need to sort file names in training directory
        this_sample=0
        for index in range(no_cumul_samples[station],no_cumul_samples[station+1]) :
            while (dir_files[this_sample]=='Thumbs.db') :
                this_sample+=1
            filename_decomposed=dir_files[this_sample].split("-")
            file_day=day_of_year[int(filename_decomposed[1])-1]+int(filename_decomposed[2])
            if file_day<day_stage[station][0] : 
                file_day+=365
            for stage in range(0,no_stages[test_station_types[station]]-1)  :
                if (day_stage[station][stage]<=file_day)and(file_day<day_stage[station][stage+1]) :
                    y_test[no_samples]=stage+(file_day-day_stage[station][stage])/(day_stage[station][stage+1]-day_stage[station][stage])
                    if (normalize_stages)   :
                        y_test[no_samples]=y_test[no_samples]*max_no_stages/(no_stages[test_station_types[station]]-1)
                    break
            if (y_train[no_samples]==0) :
                if (day_stage[station][no_stages[test_station_types[station]]-1]<=file_day) :
                   y_test[no_samples]=no_stages[test_station_types[station]]-1
                   if (normalize_stages)   :
                        y_test[no_samples]=y_test[no_samples]*max_no_stages/(no_stages[test_station_types[station]]-1)
            no_samples+=1
            this_sample+=1
# Send batches to network directly from disk via paths stored in pandas dataframe        
    df_test = pd.DataFrame({'path': test_image_path_list,  'plant_phenology_stage': y_test})  
    validation_generator=CustomImageDataGenerator(field,df=df_test,batch_size=batch_size,dim=(2048,2048),n_channels=3)    
           
    loss_log=[]
#full model trained over batches and validation loss recorded
    for e in range(-1,no_epochs2):
        if (e>-1)  :
            print('First pass Epoch', e)
            batches = 0
            for x_batch, y_batch in train_generator:
                print('Batch {0} of {1}' .format(batches, max_no_train_samples / batch_size))
                if ((batches+1)*batch_size>max_no_train_samples) :
                    upper_limit=max_no_train_samples-batches*batch_size
                else :
                    upper_limit=batch_size
                x_inp=np.ndarray(shape=(upper_limit,target_height,target_width,3), dtype=float, order='C')
                for i in range(0,upper_limit):
   #                 plt.imshow(x_batch[i])
                    img=preprocess_input(x_batch[i])
                    img/=255.
                    if _defined_contrast_stretch:
                        img=contrast_stretching(img)
                    elif _defined_luminance_stretch:
                        img=luminance_stretching(img)
                    x_inp[i] = resize(img, (target_height, target_width), anti_aliasing=True)
                    
                full_model.fit(x_inp, y_batch)
                batches += 1
                if batches >= max_no_train_samples / batch_size:
                    break
            avg_loss=0
            avg_acc=0 #accuracy is "mean absolute distance" here
            batches = 0
            for x_batch, y_batch in validation_generator:
                if ((batches+1)*batch_size>max_no_test_samples) :
                   upper_limit=max_no_test_samples-batches*batch_size
                else :
                    upper_limit=batch_size
                x_inp=np.ndarray(shape=(upper_limit,target_height,target_width,3), dtype=float, order='C')
                for i in range(0,upper_limit):
                    img=preprocess_input(x_batch[i])
                    img/=255.
                    if _defined_contrast_stretch:
                        img=contrast_stretching(img)
                    elif _defined_luminance_stretch:
                        img=luminance_stretching(img)
                    x_inp[i] = resize(img, (target_height, target_width), anti_aliasing=True)
                    
                full_model.predict(x_inp,batch_size) #output_predict_array for debug purposes
                [val_loss,val_acc]=full_model.evaluate(x_inp,y_batch,batch_size)
                batches += 1
              
                avg_acc+=val_acc*upper_limit
                avg_loss+=val_loss*upper_limit
                if batches >= max_no_test_samples / batch_size:
                    break
#full model performance averaged and recorded          
            avg_acc/=max_no_test_samples
            avg_loss/=max_no_test_samples
            loss_log.append(avg_loss)
#feature vectors predicted by final truncated model and written to disk  for train and test stations            
            fine_tune="_fine_tune_phenology/fold_"+str(fold_no)+"_"+str(e)+"/"
        else :
            fine_tune="_fine_tune_phenology/fold_"+str(fold_no)+"_initial"+"/"
        if (is_save_weights_folds)and(e!=4)  :
            if (e==-1):
                str1="_initial"
            else :
                str1="_"+str(e)
            truncated_model.save_weights('phenology_frame_classifier_model_weights_feature_epoch_no'+str1+append_str+'_fold'+str(fold_no)+'.h5')    
        if (e!=5) : # comment these two lines out if you want to save feature vector outputs of other epochs of finetuning
            continue
        for station in range(0,no_train_stations) :
            img_dataDir = "../../keras_cropped_data/"+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
        
            if (network_type=="InceptionResnetV2") :  
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                 feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                 feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+train_station_types[station]+"/"+train_station_codes[station]+"/10x/"
            if (not os.path.isdir(feature_dataDir)):
                try:  
                    os.makedirs(feature_dataDir)
                except OSError:  
                    print ("Creation of the directory %s failed" % feature_dataDir)
                    raise
                else:  
                    print ("Successfully created the directory %s " % feature_dataDir)
                    pass
            for file in train_dir_files[station] :
                if (file=='Thumbs.db') :
                    continue
                img_path = img_dataDir+file
                img = image.load_img(img_path, target_size=(target_height, target_width))
                x = image.img_to_array(img) 
                x = preprocess_input(x)
                x = x / 255.
                if _defined_contrast_stretch:
                    x=contrast_stretching(x)
                elif _defined_luminance_stretch:
                    x=luminance_stretching(x)
                    
                x = resize(x, (target_height, target_width,3), order=3, anti_aliasing=True)
                x = np.expand_dims(x, axis=0)
                y_pic=truncated_model.predict(x,batch_size=1) #feature vectors output by finetuned truncated VGG16
                base_name=file.split(".")[0]+file.split(".")[1]
                feature_path=feature_dataDir+base_name
                np.save(feature_path,y_pic)
                
        for station in range(0,no_test_stations) :
            img_dataDir = "../../keras_cropped_data/"+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
        
            if (network_type=="Resnet") :  
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_Inception_ResNet_V2"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="VGG16") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_VGG16_new"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="VGG19") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_VGG19"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="ResNet101") :
                feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_ResNet101"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="EfficientNetB7") :
                 feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_EfficientNetB7"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            elif (network_type=="DenseNet201") :
                 feature_dataDir = "../../bap_proje/comp_nets/CNNFeat_DenseNet201"+append_str+fine_tune+test_station_types[station]+"/"+test_station_codes[station]+"/10x/"
            
            if (not os.path.isdir(feature_dataDir)):
                try:  
                    os.makedirs(feature_dataDir)
                except OSError:  
                    print ("Creation of the directory %s failed" % feature_dataDir)
                    raise
                else:  
                    print ("Successfully created the directory %s " % feature_dataDir)
                    pass
            for file in test_dir_files[station] :
                if (file=='Thumbs.db') :
                    continue
                img_path = img_dataDir+file
                img = image.load_img(img_path, target_size=(target_height, target_width))
                x = image.img_to_array(img) 
                x = preprocess_input(x)
                x = x / 255.
                if _defined_contrast_stretch:
                    x=contrast_stretching(x)
                elif _defined_luminance_stretch:
                    x=luminance_stretching(x)
                    
                x = resize(x, (target_height, target_width,3), order=3, anti_aliasing=True)
                x = np.expand_dims(x, axis=0)
                y_pic=truncated_model.predict(x,batch_size=1) #feature vectors output by finetuned truncated VGG
                base_name=file.split(".")[0]+file.split(".")[1]
                feature_path=feature_dataDir+base_name
                np.save(feature_path,y_pic)
    K.clear_session() 
    return loss_log
        
is_save_weights_folds=False
_defined_contrast_stretch=True
_defined_luminance_stretch=False
if (_defined_contrast_stretch==True)  :
        append_str='contrast_stretch'
elif (_defined_luminance_stretch==True)  :

        append_str='luminance_stretch'
else :
        append_str=''
        
network_type="DenseNet201"
#network_type="ResNet101"
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
    data_dim=1536
elif (network_type=="EfficientNetB7") :
    load_data_type="python"
    data_dim=1536
elif (network_type=="DenseNet201") :
    load_data_type="python"
    data_dim=94080
    
if (network_type=="InceptionResnetV2") :
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    target_height=224
    target_width=224
elif (network_type=="VGG16") :
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    target_height=224
    target_width=224
elif (network_type=="VGG19") :
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    target_height=224
    target_width=224
elif (network_type=="ResNet101") :
    from tensorflow.keras.applications.resnet import ResNet101
    from tensorflow.keras.applications.resnet import preprocess_input
    target_height=224
    target_width=224
elif (network_type=="EfficientNetB7") :
    from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
    from tensorflow.python.keras.applications.efficientnet import preprocess_input
    target_height=600
    target_width=600
elif (network_type=="DenseNet201") :
    from tensorflow.python.keras.applications.densenet import DenseNet201
    from tensorflow.python.keras.applications.densenet import preprocess_input
    target_height=224
    target_width=224      
   

fold_no_start=int(sys.argv[1])    
fold_no_end=int(sys.argv[2]) 

normalize_stages=1
batch_size=64
no_epochs2=6 

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

train_stations=np.loadtxt("train-partition",dtype='int')
test_stations=np.loadtxt("test-partition",dtype='int')

for fold_no in range(fold_no_start,fold_no_end) :
    loss_fold=process_fold_data2(fold_no,1) 
    for ep in range(0,no_epochs2) :
        print("Epoch {0} Fold {1} Avg. acc {2}".format(ep,fold_no,loss_fold[ep]))


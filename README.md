The python source files and dataset are associated with the following paper:

"Classifying types and predicting phenological stages of crops from in situ image sequences by deep learning", 
U. Bayazit, D.T. Altilar, N.G. Bayazit
Turkish Journal of Electrical Engineering and Computer Sciences, 2022.

If you use any of the source code files in your own research work and make a publication, please cite this work.

The files 
plant_type_classifier_v3_kfold_cross_val_deepnets.py 
and 
plant_phenology_classifier_v3_kfold_cross_val_deepnets.py 
are used to design the first stage feature extractor networks.

The files 
plant_type_classifier_v1_kfold_cross_val_deepnets_tf1.py 
and 
plant_phenology_classifier_v1_kfold_cross_val_deepnets_tf1.py 
are used to design the second stage recurrent networks that operate on the feature vector outputs of the corresponding first stage network.

The file 
CustomizedImageDataGenerator_tf2  
generates batches of images for training/validating the first stage feature extractor network.

The files 
batch_generator_phenology.py 
and 
batch_generator.py 
generate batches of windows of feature vectors for training/validating the recurrent networks used in classification/prediction. 

Important note about Tensorflow version: Even though the results reported for VGG-16 in the article referenced above are based on Tensorflow 1.12 only, for comparison with the deep networks of ResNet101 and DenseNet201, the original source codes for the feature extractor networks and their batch generator (not provided here) have been ported to Tensorflow 2.6. The changes, however, are only minor.

Note about the sample dataset: A sample dataset (watermarked) has been provided to give an idea about the kind of image sequence data processed. The sample size is approximately 1/10th of the dataset used in the simulations of the above mentioned paper.

If you wish to gain access to the original dataset used in the studies, please contact Prof. U. Bayazit (ulugbayazit@itu.edu.tr) or Prof. T. Altilar (altilar@itu.edu.tr) for licensing. Istanbul Technical University and Ministry of Agriculture and Forestry of Turkey have exclusive rights on the dataset.
The dataset comes together with metadata for each plant type that, most importantly, lists the onset dates of each of the phenological stages. The metadata files are accessed from the source codes, and therefore, should not be changed for proper stage labelling.
This project supplies the python source files and dataset used in the following paper:
"Classifying types and predicting phenological stages of crops from in situ image sequences by deep learning", 
U. Bayazit, D.T. Altilar, N.G. Bayazit
Turkish Journal of Electrical Engineering and Computer Sciences, 2022.

Please cite this work, if you use any of the source code files in your own research work and make a publication.

The files plant_type_classifier_v3_kfold_cross_val_deepnets.py and plant_phenology_classifier_v3_kfold_cross_val_deepnets.py 
are used to design the first stage feature extractor networks.
The files plant_type_classifier_v1_kfold_cross_val_deepnets_tf1.py and plant_phenology_classifier_v1_kfold_cross_val_deepnets_tf1.py 
are used to design the second stage recurrent networks that operate on the feature vector outputs of the corresponding first stage network.

Important note: Even though the results reported for VGG-16 in the article referenced above are based on Tensorflow 1.12 only, for comparison with the deep netowrks of ResNet101 and 
DenseNet201, the original source codes for the feature extractor networks (not provided here) have been ported to Tensorflow 2.6. The changes are only minor.

The files batch_generator_phenology.py and ba
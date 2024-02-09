# AIAQUAMI Training Scripts
Training scripts for the AIAQUAMI project

## Prerequisites
- Training process relies on TensorFlow 2.9.3. 
- To be able to use TensorFlow 2.9.3 on GPU you must first install CUDA 11.2 and cuDNN 11.2.
- After setting up CUDA and cuDNN, install project requirements listed in <code>[requirements.txt](requirements.txt)</code> file.

## Train model on a custom dataset
1) Create root folder for the dataset and place images into <code>data/data_original</code> subfolder. 
Images for each class should be placed in a separate folder. Name of these folders should correspond to class names.
2) In <code>[settings.py](settings.py)</code> change value for <code>root_folder</code> variable to appropriate value.
3) Reset <code>experiment_no</code> to 1 and check other parameters available in <code>[settings.py](settings.py)</code>.
4) Run <code>[prepare_data.py](prepare_data.py)</code> script to resize images and divide them into train, validation, and test subsets.
5) Run <code>[train.py](train.py)</code> to train the model, generate confusion matrices and Grad-CAM heatmaps. 
Results will be saved to <code>tmp</code> dataset subfolder. 
6) In case training crushes due to lack of memory, reduce <code>batch_size</code> or choose smaller model and repeat training.
7) Once training is finished, you may find results in <code>tmp</code> folder, under the current experiment subfolder. 

## Download pretrained models
Pretrained models are available for download in the repository release section:<br/>
https://github.com/a-milosavljevic/aiaquami-training/releases

## Acknowledgement
This research was supported by the [Science Fund of the Republic of Serbia](http://fondzanauku.gov.rs/?lang=en), #7751676, Application of deep learning in bioassessment of aquatic ecosystems: toward the construction of automatic identifier of aquatic macroinvertebrates - [AIAQUAMI](https://twitter.com/AIAQUAMI).

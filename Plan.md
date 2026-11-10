# Plan

## Ideal pipeline
1. Microsocopic organism images are taken and labelled with proper name and magnification level
2. Bounding boxes are automatically calculated using OpenCV
import cv2
3. Use ML on bounded box images to detect organisms

## Future Avenues

This is in reference to the advertisement numbered PU/507/ADV/899 for the position of PA-I in the project of "Machine learning for Image-based identification of nematodes - a novel paradigm in biodiversity assessment and precision agriculture".

I went through your research "A deep learning framework for species-level identification of root-knot nematodes using perineal pattern images" where you've previously conducted nematode classification using AlexNet and VGG16. I also went through the supplementary material to have a look at the model infrastructures, also to realize that this domain is pretty novel and underexplored. I would be thrilled to contribute to such an emergent field of research.

Gone through the research findings and discussions for the future, and the following possible extensions come to my mind:
1. Models that classify based jointly on morphological measurements as well as image characteristics. Ensemble methods?
2. Soil health and agricultural plausibility analysis through automated nematode-detection pipelines.
3. Extending the analysis to more varieties of nematodes from the previously explored, most common 3 varieties.
4. Extending classification into another scheme (for example, you mention trophic groups).
5. Futher research on explaining model performance.

# My pipeline

KISS: keep it short simple. dont make it verbose and elaborate and convoluted and full of bullshit loops and stuff. make code readable.
1. rotate every image by some random angle and create 5 copies for each such image in the train folder. keep the val folder intact
2. train and test over the train folder. once model is trained, validate on the val folder by running the model on it.
3. download the models separately without any if-else stuff, in some directory and accessed from there, instead of a cache that goes away every kernel restart (right?). write one code block that downloads it if the model storage folder is empty.
4. run the 3 models without for loops in 3 separate code blocks so that i can run the second one without rerunning the first one.
5. once models are fully trained, save the trained model files somewhere else than the initial files, so that i dont have to rerun model training everytime.
6. let model be downloaded 
7. download alexnet, vgg16 and resnet 50 and use all 3 and compare them
8. write code for model explainability using a methodology you think is right for this context

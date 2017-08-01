# Gender-Image-Classifier
Uses Keras with a TensorFlow Backend to classify facial photos as male or female.

To use this to create your own model you would do a few steps:

  -  Download lots of facial images - I used the LFW dataset: http://vis-www.cs.umass.edu/lfw/

  -  run the image_classifier.py script

      -  This will write your model to a JSON file and the weights for your model to an h5 file.
  
This can be used to classify images by:

  -  Importing your model and load your weights

  -  read images you'd like to classify into numpy array

  -  Call the 'model.predict()' function on your numpy array representation of the image you'd like to classify.

An example of this can be found in the testing.py file.

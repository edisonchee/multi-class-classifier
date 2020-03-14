# Multi-class Image Classifier

Transfer Learning using MobileNet V2 with real-time image augmentation.

Python version: `3.7.6`

## Getting started

Setup your environment. I recommend pipenv:
```sh
pipenv shell
pipenv install numpy matplotlib tensorflow 
```

Create necessary folders:
```sh
mkdir train validate checkpoint
```

Organise your images into folders in both `train` and `validate` folders:
```
train
  |-poodle
    |-poodle_0.jpg
    |-poodle_1.jpg
    |-...
  |-pomeranian
    |-pomeranian_0.jpg
    |-pomeranian_1.jpg
    |-...
  |-shiba_inu
    |-shiba_0.jpg
    |-shiba_2.jpg
    |-...
  |-...
validate
  |-poodle
    |-...
  |-pomeranian
    |-...
  |-shiba_inu
    |-...
  |-...
```

Classes are automatically generated by [ImageDataGenerator.flow_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator).
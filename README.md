# kaggle_TAU_competition
competition on the machine learning course.
[Kaggle competition] (https://www.kaggle.com/c/vehicle/overview/) - web-page of the competition.

Our group consisting of me, Eetu Nisula, Lauri Hautaniemi ja Raimo Yli-Peltola was part of
competition. The task was to classify vehicle types using CNN and other techniques.
Our group has 21. place on the leaderboard mostly because of lack of computational capacity.

* alg_test.py - here I gathered a bunch of "ordinary" classifiers and tried to pass training data throug them. That gave me 65 - 75% accuracy
* fst_try.py - VGG16 network without top layers is trained giving about 85% accuracy on training data
* vgg_again_train.py - played with fine tuning of VGG16 network
* pretrain.py - MobileNet without top layers was used as feature extractor
* inception_model.py - inception network was trained. Afterwards this model was trained further by setting as trainable several convolutional layers. The code is in the folder CheckPoints/inception_model.py




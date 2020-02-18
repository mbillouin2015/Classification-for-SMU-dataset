# Classification For The SMU Dataset

This project proposes three Neural Network Architectures for a binary classification task and comparing their performance for a given dataset. The models include a Multi-Layer Perceptron (MLP), a Long-Short Term Memory Neural Network (LSTM), and a Convolutional Neural Network (CNN). The scope of the project is to discriminate between events that produce infrasound signals, low-frequency acoustic waves less than 20 Hz. During the period 2003 – 2013, ground truth infrasonic arrivals were documented on the explosive disposal of rocket motors at the Utah Test and Training Range (UTTR) and extended rocket motor burn tests (RMT) of horizontally-oriented rockets at Promontory, Utah, studied at the Southern Methodist University [1]. These two experiments will serve as the two classes in this binary classification problem. The data from these two experiments will be obtained through queries to the Incorporated Research Institutions for Seismology (IRIS) system, ensuring that the data exists in the public domain. This dataset will henceforth be known as the SMU dataset. Minimal preprocessing of each signal is preferred to reduce computational overhead. With this in mind, evaluations of the deep learning architectures proposed will be performed on two versions of the dataset:

- Raw: The signals will be evaluated as queried from IRIS with no preprocessing applied.
- Resampled: All the signals in the dataset will be resampled using the Fourier method to a common sampling rate, between the maximum and minimum of the dataset. With the SMU data, this sampling rate is 200 Hz.

The following list illustrates the tasks that need to be completed:
1.	Data Acquisition – The SMU data will be queried from IRIS using a pythonic interface using the package ObsPy [2].
2.	Data Cleaning and Preparation – To yield better classification results, the data used to train the models should have a Signal To Noise (SNR) ratio as low as possible. As a result of this fact, signals with no clear events (a high SNR) will be removed from the dataset. This new curated dataset will be used for classification. The signals will also be resampled as explained above so there will be two versions of the dataset. The data will be split into a training set (70% of the data), used to train the model and a testing set (30% of the data) for testing the generalizability of the models.
3.	Model Selection and Training – In this phase, each model proposed (MLP, CNN, and LSTM) will be trained on the training set of both versions of the SMU data. This will include the designing the model architectures and hyperparameter selections as applicable. 
4.	Model Testing – Here, each model is evaluated on the testing set for both versions of the SMU dataset. 
5.	Model Comparisons – All classification metrics are based on model performance on the test set true labels versus the predicted labels by each classifier. 

The expected outcome of this project is to have models that can discriminate between events that produce infrasonic signals with high fidelity and to determine the champion deep learning mo del (model with the highest fidelity) from the models proposed.


**References:**

[1] J. Park, C. Hayward, and B. W. Stump, “Assessment of infrasound signals recorded on seismic stations and infrasound arrays in the western United States using ground truth sources,” Geophys. J. Int., 2018.

[2] M. Beyreuther, R. Barsch, L. Krischer, T. Megies, Y. Behr and J. Wassermann (2010)
ObsPy: A Python Toolbox for Seismology, SRL, 81(3), 530-533, DOI: 10.1785/gssrl.81.3.530

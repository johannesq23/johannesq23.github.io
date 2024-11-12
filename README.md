# Directories/Files:

/dir/: Root folder for all other files and folders for the project 

/dir/data/: Folder that stores the downloaded .txt data files

/dir/models.py: Intializes and defines run methods for ML models

/dir/requirements.txt: Defines package/library dependencies 

/dir/testing.ipynb: Jupyter notebook that includes data cleaning, feature engineering/reduction, and training/testing the ML models on the cleaned data frame

# Introduction/Background:

### Our Data Set
https://huggingface.co/datasets/DavidVivancos/MindBigData2022


Interpreting brain signals as responses to visual stimuli is an exciting topic of research, with a wide variety of applications in healthcare [1], education [2], and entertainment [3]. These signals can be easily obtained using electroencephalograms (EEG), which employ signal processing techniques like Fourier transforms and spectral analysis to generate meaningful interpretations [4]. Numerical digits are commonly chosen as stimuli in this research because they are discrete, limited in number (0-9), and universally understood [5, 6].

Our data set was developed by David Vivancos, who used 4 different EEG machines to track activity in 19 sections of his own brain upon being shown an image of a single digit at a time. These images ranged from 0-9, or no digit as a control. The dataset includes 4 main sub-datasets for each EEG machine used. For each subset, there exists a “digit” feature corresponding to the digit shown, an “event id” to catalog unique number-showing events, a “brain region” to map the location from where an electrode made its reading, and ~250 columns of time series data sampled at 128 Hz that correspond to the electric intensities measured from the given “brain region”. Therefore, for our purposes, a datapoint can be considered all rows which share the same “event id” which is 19 for the full data set and 5 for the exploratory data.


### Problem Definition
We seek to develop a more accurate system of decoding brain signals associated with specific visual stimuli, in this case related to numerical digits. Through our model, it will be possible to predict what digit is being seen by the candidate given their brain activity.
This research will be applicable in proving the feasibility of brain-computer interfaces (BCI’s), which have immense potential for improving quality of life for individuals experiencing disabilities in physical or verbal communication. When methods such as typing, speaking, or gesture-based systems are inaccessible, these BCIs that solely rely on brain activity could minimize the accessibility gap.

### Methods
Because the electrode data is formatted as a time series, the feature columns did not directly correspond to a measurable feature. Thus, classical feature reduction techniques such as PCA would not be helpful in determining which “time slices” are most valuable to analyze. Conversely, it would make more sense to consider all “time slices” from a given electrode reading during an event. Therefore, we preprocessed the data by reducing each time series to its core summary statistics, max, min, mean, and range. Following, because a datapoint can be characterized as all rows that share an “event id”, we further preprocessed the data by concatenating rows from different brain regions that share an “event id.” 

Post preprocessing, we determined a datapoint to be composed of the max, min, mean, and range for each “brain region” for a given “event id.” We then trained Logistic Regression and Random Forest supervised learning models on the preprocessed data.

### Results and Discussion

# Logistic Regression:
Accuracy Score: 11.15%
F1 Score: 9.79%

![logistic](https://github.com/johannesq23/johannesq23.github.io/blob/main/Logistic%20Regression%20Confusion%20Matrix.png)

# Random Forest:
Accuracy: 9.77%
F1 Score: 9.72%

![randomforest](https://github.com/johannesq23/johannesq23.github.io/blob/main/Random%20Forest%20Confusion%20Matrix.png)

We tested our data on two supervised classifiers: Logistic Regression and Random Forest. Logistic regression was our initial baseline to see how the data might behave with a classifier. Based on the confusion matrix, logistic regression tended to guess certain values at a far greater frequency, classifying most data points as either a 1, 4, 7, or 9. We then wanted to try something more complex that could handle potentially non-linearly separable data so we used a Random Forest Classifier. For this case, it was very clear that the model was guessing given a more even distribution of predictions in the confusion matrix.

The performance of our models fell significantly short of expectations, with accuracy and F1 scores barely exceeding random guessing. The Logistic Regression model achieved an accuracy of 11.15% and an F1 score of 9.79%, while the Random Forest model yielded 9.77% accuracy and a 9.72% F1 score, both hovering around the baseline for random classification among 10 classes. These results suggest that the models were essentially guessing, with no significant pattern discerned in the predictions.

These models were limited by the overly simplified representation of EEG signals. Reducing each time series to basic summary statistics (max, min, mean, range) likely discarded essential temporal information, which is crucial in brainwave analysis. 

In the future, we plan to make several enhancements. First, we will incorporate Fourier Transforms to shift EEG data into the frequency domain, allowing us to capture patterns in specific frequency bands associated with cognitive states. By applying PCA on these Fourier-transformed features, we hope to retain important patterns while reducing dimensionality. Additionally, we plan to explore Recurrent Neural Networks (RNNs) or Long Short-Term Memory networks (LSTMs) as they are designed for time-series data and can capture sequential dependencies, as logistic regression and random forest are used for independent data points.

# Gantt Chart:

![gantt](http://johannesq23.github.io/gantt.png)

# Contribution Chart:

![contribution](http://johannesq23.github.io/contributions.png)

# References:

[1] X. Chai, T. Cao, Q. He, N. Wang, X. Zhang, X. Shan, Z. Lv, W. Tu, Y. Yang, and J. Zhao, "Brain-computer interface digital prescription for neurological disorders," CNS Neuroscience & Therapeutics, vol. 30, no. 2, pp. e14615, Feb. 2024. doi: 10.1111/cns.14615. PMID: 38358054; PMCID: PMC10867871.

[2] P.-C. Hu and P.-C. Kuo, "Adaptive learning system for e-learning based on EEG brain signals," in 2017 IEEE 6th Global Conference on Consumer Electronics (GCCE), Nagoya, Japan, 2017, pp. 1-2, doi: 10.1109/GCCE.2017.8229382.

[3] D. de Queiroz Cavalcanti, F. Melo, T. Silva, M. Falcão, M. Cavalcanti, and V. Becker, "Research on brain-computer interfaces in the entertainment field," in Human-Computer Interaction, M. Kurosu and A. Hashizume, Eds. Cham, Switzerland: Springer, 2023, vol. 14011, pp. 1-10. doi: 10.1007/978-3-031-35596-7_26.

[4] M. Sokač, L. Mršić, M. Balković, and M. Brkljačić, “Bridging Artificial Intelligence and Neurological Signals (brains): A novel framework for electroencephalogram-based image generation,” Information, vol. 15, no. 7, p. 405, Jul. 2024. doi: 10.3390/info15070405.

[5] S. Tiwari, S. Goel, and A. Bhardwaj, "EEG Signals to Digit Classification Using Deep Learning-Based One-Dimensional Convolutional Neural Network," Arabian Journal for Science and Engineering, vol. 48, pp. 9675–9691, 2023, doi: 10.1007/s13369-022-07313-3.

[6] N. C. Mahapatra and P. Bhuyan, "EEG-based classification of imagined digits using a recurrent neural network," Journal of Neural Engineering, vol. 20, no. 2, p. 026040, Apr. 2023, doi: 10.1088/1741-2552/acc976.

[7] Scikit-learn developers, "sklearn.preprocessing.MinMaxScaler," Scikit-learn, 2023. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

[8] Scikit-learn developers, "sklearn.decomposition.PCA," Scikit-learn, 2023. [Online]. Available: https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html

[9] M. Adel, "Tomek Links," Medium, May 30, 2023. [Online]. Available: https://medium.com/@mahmoudadel200215/tomek-links-948ea097199e

[10] Ibm. (2024, August 23). What is Random Forest?. IBM. https://www.ibm.com/topics/random-forest. 

[11] What is XGBoost?. NVIDIA Data Science Glossary. (n.d.). https://www.nvidia.com/en-us/glossary/xgboost/ 

[12] GeeksforGeeks. (2023, January 30). Support vector regression (SVR) using linear and non-linear kernels in Scikit learn. https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/ 



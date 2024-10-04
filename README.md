# Introduction/Background (196 words)
### Our Data Set: https://huggingface.co/datasets/DavidVivancos/MindBigData2022
Interpreting brain signals as responses to visual stimuli is an exciting topic of research, with a wide variety of applications in healthcare [1], education [2], and entertainment [3]. These signals can be easily obtained using electroencephalograms (EEG), which employ signal processing techniques like Fourier transforms and spectral analysis to generate meaningful interpretations [4]. Numerical digits are commonly chosen as stimuli in this research because they are discrete, limited in number (0-9), and universally understood [5, 6].
Our data set was developed by David Vivancos, who used 4 different EEG machines to track activity in 19 sections of his own brain upon being shown an image of a single digit at a time. These images ranged from 0-9, or no digit as a control. The dataset includes 4 main sub-datasets for each EEG machine used. Within each subset, there is a “digit” feature corresponding to the digit shown, and all other features are brain activity trackers of various parts of the 19 sections of the brain. There are approximately 100-1,000 tracking channels (each its own feature/column) for each brain region or electrode site (i.e. right frontal region), with at least 10 different electrode sites being monitored.
Problem Definition (104 words)
We seek to address the development of a more accurate system of decoding brain signals associated with specific visual stimuli, in this case related to numerical digits. Through our model, it will be possible to predict what digit is being seen by the candidate when given their brain activity.
This research will be applicable in proving the feasibility of brain-computer interfaces (BCI’s), which have immense potential for improving quality of life for individuals experiencing disabilities in physical or verbal communication. When methods such as typing, speaking, or gesture-based systems are inaccessible, these BCIs that solely rely on brain activity could minimize the accessibility gap.



# Proposed Solution (200 words):

To predict numbers from EEG readings, an effective solution involves a combination of preprocessing techniques and machine learning models.

### Data Preprocessing Methods:
Feature Scaling: EEG data can have varying amplitude ranges, so normalization (e.g., MinMaxScaler from scikit-learn) is essential for standardizing the data. [1]
Dimensionality Reduction: EEG data often has high dimensionality. Applying Principal Component Analysis (PCA) (PCA from scikit-learn) can reduce noise and prevent overfitting while maintaining key variance. [2]
Tomek Undersampling: Reducing the frequency of noisy data that does not pertain to detecting digits or dealing with a class imbalance for one particular digit. [3]

[1] Scikit-learn developers, "sklearn.preprocessing.MinMaxScaler," Scikit-learn, 2023. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
[2] Scikit-learn developers, "sklearn.decomposition.PCA," Scikit-learn, 2023. [Online]. Available: https://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html
[3] M. Adel, "Tomek Links," Medium, May 30, 2023. [Online]. Available: https://medium.com/@mahmoudadel200215/tomek-links-948ea097199e



## Machine Learning Algorithms:
Since this dataset comes with labels, this will be supervised learning rather than unsupervised learning. There are a variety of classification algorithms that we can use in the scikit-learn library which will meet our needs.
Random Forest Regressor: This method essentially creates a multitude of decision trees that all come to their own conclusion, where the results are chosen based on majority[7]. This method handles non-linearities well and is resistant to overfitting, making it ideal for complex EEG data with high dimensionalities. 
XGBoost Regressor: XGBoost adds on smaller learners sequentially, reducing the error down in steps/gradient. XGBoost is efficient and can handle large datasets with high accuracy. However, tuning models such as XGBoost can be tricker, with the number of hyperparameters.[8]
Support Vector Regression: 
This model searches for a hyperplane to fit the majority of the points. This model helps against outliers, which may be effective in the EEG’s/brain’s random activity, which makes it a suitable candidate for this project [9].




#Potential Results and Discussion:

Potential Results and Discussion (117 words)

Foremost, we seek to design a high efficacy model. Because our problem manifests as classification, accuracy can be ensured by maximizing accuracy score and top-k accuracy score, which concurrently boosts usability. Similarly, considering the use case of the model in consequential environments, mitigating false positives and negatives can be ensured by maximizing precision, recall, and F1 scores.

From a sustainability standpoint, we seek to design a light-weight model: minimizing model size reduces energy expenditure. Ethically, we seek for the model to be generalizable; not available to just those with resources to acquire custom training data.

We expect to develop a high efficacy ( > 85% accuracy and > 85% F1 Macro) and lightweight model that generalizes to the entire population.

Calculate per class accuracy based on the digit

#References:

[1] X. Chai, T. Cao, Q. He, N. Wang, X. Zhang, X. Shan, Z. Lv, W. Tu, Y. Yang, and J. Zhao, "Brain-computer interface digital prescription for neurological disorders," CNS Neuroscience & Therapeutics, vol. 30, no. 2, pp. e14615, Feb. 2024. doi: 10.1111/cns.14615. PMID: 38358054; PMCID: PMC10867871.
[2] P.-C. Hu and P.-C. Kuo, "Adaptive learning system for e-learning based on EEG brain signals," in 2017 IEEE 6th Global Conference on Consumer Electronics (GCCE), Nagoya, Japan, 2017, pp. 1-2, doi: 10.1109/GCCE.2017.8229382.
[3] D. de Queiroz Cavalcanti, F. Melo, T. Silva, M. Falcão, M. Cavalcanti, and V. Becker, "Research on brain-computer interfaces in the entertainment field," in Human-Computer Interaction, M. Kurosu and A. Hashizume, Eds. Cham, Switzerland: Springer, 2023, vol. 14011, pp. 1-10. doi: 10.1007/978-3-031-35596-7_26.
[4] M. Sokač, L. Mršić, M. Balković, and M. Brkljačić, “Bridging Artificial Intelligence and Neurological Signals (brains): A novel framework for electroencephalogram-based image generation,” Information, vol. 15, no. 7, p. 405, Jul. 2024. doi: 10.3390/info15070405.
[5] S. Tiwari, S. Goel, and A. Bhardwaj, "EEG Signals to Digit Classification Using Deep Learning-Based One-Dimensional Convolutional Neural Network," Arabian Journal for Science and Engineering, vol. 48, pp. 9675–9691, 2023, doi: 10.1007/s13369-022-07313-3.
[6] N. C. Mahapatra and P. Bhuyan, "EEG-based classification of imagined digits using a recurrent neural network," Journal of Neural Engineering, vol. 20, no. 2, p. 026040, Apr. 2023, doi: 10.1088/1741-2552/acc976.
[7] Ibm. (2024, August 23). What is Random Forest?. IBM. https://www.ibm.com/topics/random-forest. 
[8] What is XGBoost?. NVIDIA Data Science Glossary. (n.d.). https://www.nvidia.com/en-us/glossary/xgboost/ 
[9] GeeksforGeeks. (2023, January 30). Support vector regression (SVR) using linear and non-linear kernels in Scikit learn. https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/ 

# Introduction/Background:

### Our Data Set: https://huggingface.co/datasets/DavidVivancos/MindBigData2022 
Interpreting brain signals as responses to visual stimuli is an exciting topic of research, with a wide variety of applications in healthcare [1], education [2], and entertainment [3]. These signals can be easily obtained using electroencephalograms (EEG), which employ signal processing techniques like Fourier transforms and spectral analysis to generate meaningful interpretations [4]. Numerical digits are commonly chosen as stimuli in this research because they are discrete, limited in number (0-9), and universally understood [5, 6].

Our data set was developed by David Vivancos, who used 4 different EEG machines to track activity in 19 sections of his own brain upon being shown an image of a single digit at a time. These images ranged from 0-9, or no digit as a control. The dataset includes 4 main sub-datasets for each EEG machine used. Within each subset, there is a “digit” feature corresponding to the digit shown, and all other features are brain activity trackers of various parts of the 19 sections of the brain. There are approximately 100-1,000 tracking channels (each its own feature/column) for each brain region or electrode site (i.e. right frontal region), with at least 10 different electrode sites being monitored.

### Problem Definition
We seek to develop a more accurate system of decoding brain signals associated with specific visual stimuli, in this case related to numerical digits. Through our model, it will be possible to predict what digit is being seen by the candidate given their brain activity.
This research will be applicable in proving the feasibility of brain-computer interfaces (BCI’s), which have immense potential for improving quality of life for individuals experiencing disabilities in physical or verbal communication. When methods such as typing, speaking, or gesture-based systems are inaccessible, these BCIs that solely rely on brain activity could minimize the accessibility gap.



# Proposed Solution:

### Data Preprocessing Methods:
1. Feature Scaling: EEG data has varying amplitude ranges, so normalization (e.g., MinMaxScaler from scikit-learn) is essential for standardization. [7]
2. Dimensionality Reduction: EEG data often has high dimensionality. Applying Principal Component Analysis (PCA from scikit-learn) reduces noise and prevents overfitting while maintaining key variance. [8]
3. Aims to reduce the frequency of noisy data that does not pertain to classifying digits or dealing with a class imbalance for one particular digit. [9]


### Machine Learning Algorithms:
Because the dataset features labels that enable supervised learning, some key classification algorithms include:
1. Random Forest Generator: Creates a multitude of decision trees that all form their own conclusion, where results are chosen based on majority [10]. The algorithm handles non-linearities well and resists overfitting, making it ideal for complex EEG data with high dimensionalities.
2. XGBoost Regressor: Adds on smaller learners sequentially, reducing the error in steps. XGBoost is efficient and can handle large datasets with high accuracy. However, tuning models such as XGBoost can be trickier, with the number of hyperparameters.[11]
3. Support Vector Regression: Searches for a hyperplane to fit the majority of the points. It is strong against outliers, an effective attribute due to the EEG’s/brain’s random activity [12].



# Potential Results and Discussion:

Foremost, we seek to design a high efficacy model. Because our problem manifests as classification, accuracy can be ensured by maximizing accuracy score (both per-class and overall) and top-k accuracy score (all > 85%), which boosts usability. Similarly, considering the use case in high-impact environments, mitigating false positives and negatives can be ensured by maximizing precision, recall, and F1 scores (all > 85%).

From a sustainability standpoint, we seek to design a light-weight model: minimizing model size reduces energy expenditure. Ethically, we seek for the model to be generalizable; not available to just those with resources to acquire custom training data.

# Gantt Chart:

![gantt](http://johannesq23.github.io/gantt.png)

# Contribution Chart:

![contribution](http://johannesq23.github.io/contribution.png)

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



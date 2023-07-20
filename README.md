# Hand-Movement-Classification-based-on-EMG-Signals-using-Machine-learning-and-Deep-learning

## Project Implementation
Our project was executed in two primary ways: machine learning and deep learning. Each 
pathway made a distinct contribution to our study. Upon completing the implementation, we could 
thoroughly analyse the outcomes and draw conclusions regarding which approach best complemented 
our study and yielded the most accurate results.

The overall process started with the EMG data collected from the MYO Thalmic bracelet. 
Where, each class data was recorded. Then our EMG data were processed using different methods. 
Features were extracted using various functions, and lastly, data was fed to the different ML & DL
classifiers for evaluation and testing.

The whole process including preprocessing, feature extraction, and testing was implemented 
using Python. Python was a great option due to its large sets of libraries, such as the Numpy, Pandas, 
and Matplotlib used in our machine learning experiment.


## EMG dataset:
Obtaining raw data can be very difficult and time-consuming. Fortunately, in our study, we 
were lucky enough to be shared with analytics-ready data that we can utilize effectively. To record the 
signals MYO Thalmic bracelet was used, it is designed to be worn on the user’s forearm, next to a PC 
with a Bluetooth receiver (Sojan, 2020). This armband is designed with no less than 8 sensors 
surrounding the forearm that can capture the myographic signals. Soon to be transferred to the PC 
through the Bluetooth interface.


In our research, the EMG data provided was mined from 36 subjects. Every subject was asked 
to perform several static hand gestures for two rounds. Each round held six-seven basic gestures that 
can be characterised as hand in rest, hand clenched in a fist, wrist flexion, wrist extension, radial 
deviation, ulnar deviation, and extended palm (not performed by all subjects). Every gesture was 
carried out for 3 seconds along with a 3 seconds pause gap in between gestures (Sojan, 2020). Our 
dataset consisted of 11 columns, with the first column representing time in milliseconds (ms), followed 
by columns 2-8 representing the channels (EMG channels of MYO Thalmic armband), column 10 
denoting label of gestures, and lastly, the label column referring to the subject who took over the 
gesture.


## Machine Learning-Based Classification:

We settled on 4 different supervised ML classifiers that could perform well in terms of gesture 
classification. The Support-vector machine, k-nearest Neighbours Algorithm, Decision Tree, and 
Random Forest. These classifiers were preferred for comparison, based on several performance metrics 
such as accuracy, and precision.

SVM is one of the algorithms that is frequently preferred for classification use, such as body 
movements, images, texts and handwriting, time series, and medical predictions. (Toledo-Perez et al., 
2019). It is a branch of supervised learning algorithms, that helps in minimizing structural risk and 
includes statistical learning theories (SENTURK & BAKAY, 2021). SVM mainly works by finding an 
optimal hyperplane that is separated into a highly-dimensional feature space. (Kaczmarek et al., 2019) 
This can be achieved by mapping entries using non-linear functions. This allows SVM to effectively 
differentiate between two or more types of objects. Also, SVM offers several Kernels that are 
exponential, radial, and polynomial functions. Its ability to handle non-linear functions through Kernel 
functions makes it easy to capture complex patterns in EMG data. For EMG hand gesture 
classification, polynomial is one suitable function used as Kernel.

k-NN is also a supervised learning algorithm that is widely used for classification and 
regression functions. It functions based on the principle of similarity where neighbours have 
contributions based on their distance. The most essential parameter stands to be the k, which holds the 
value of the number of neighbours (SENTURK & BAKAY, 2021). Its mechanism revolves around 
calculating the distance between the test sample and the training sample and then choosing the knearest training samples to the test samples, then finally classifying them into the class with the highest probability (AlOmari & Liu, 2014). It is common to find k-NN algorithms in the fields of pattern recognition, recommendation systems, anomaly detection, and image and text classification. Lastly, KNN provides a fluent non-parametric process for assigning class labels to input patterns based on the 
class labels represented by the k (Rasheed et al., 2006).


DT defines a supervised learning algorithm that works by taking an input, a table for example 
and recursively dividing the table into sub-tables and so improving the purity score of the column 
“label” in each partition. Therefore, the more the proportion of one class, the purer the collection stands 
to be (Yeturu, 2020). DT is constantly used to classify biomedical signals; the instances get sorted from 
the root to some leaf node, and the attributes of the instances get tested and directed to a sub-node, each 
branch denotes an outcome of the test, and each leaf holds a class label (Keleş & Subaşı, n.d). After 
checking all attributes and their values, the target values of the new instance can be defined. DT 
provides further advantages such as requiring minimal data preparation and strong performance even 
on big datasets (Kraslawski & Turunen, 2013).

RF is a classification technique that can be described as an ensemble of decision trees (Zhou et 
al., 2019). Whereas decision trees tend to know how to classify a given instance of data, random forests 
tend to bootstrap and choose the best prediction. Meaning that each tree in the random forest generates 
a class prediction, and the class that wins with the highest number of “votes” happens to become our 
model’s prediction (Kundu, 2021). Using random forests, trees protect each other from pursuing 
individual mistakes, although some trees are incorrect, many turn out right, therefore as a group, the 
trees are headed to move in the right direction. For this specific reason, uncorrelated models can 
conduct more accurate ensemble prediction, than any other individual prediction (Kundu, 2021). 

## Deep Learning-Based Classification:
Developing a deep learning-based classification system for hand and finger movement 
classification using EMG signals involves several steps. First, the data needs to be preprocessed, which 
may include filtering, normalization, and segmentation. Feature extraction is then performed to extract 
relevant information from the EMG signal. This may involve using time-domain or frequency-domain 
features, or a combination of both (Vasanthi SM, Jayasree T.). Model design is the next step, where a 
neural network architecture is selected and configured. Depending on the type of neural network, the 
architecture may include Dense layers, recurrent layers, or a combination of both. Finally, the model is 
fine-tuned using training data, and the performance of the model is evaluated using testing data(R. B. 
Azhiri, M. Esmaeili and M. Nourani,).

There are different types of neural network models that can be used for this task. Convolutional 
neural networks (CNNs) have been shown to be effective in capturing spatio-temporal patterns in EMG 
signals. However, it shows a good result when the dataset is images. Since our data set is a set of 
numbers, CNN was not a good choice. So we used an artificial neural network to classify the EMG 
signals.

Artificial neural network (ANN) is a computational model inspired by the structure and 
function of biological neurons in the brain (Christodoulou & Pattichis 1999). It consists of a large 
number of interconnected nodes or neurons that are organized into layers. Each neuron receives input 
18
signals from other neurons or external sources, performs a computation on these inputs, and produces 
an output signal that is transmitted to other neurons or output nodes (Saeed et al. 2021).

ANNs are good at working with classifying EMG signals because they can learn to extract 
useful features from complex and high-dimensional data (Aqsa Saeed Qureshi, Asifullah Khan, Aneela 
Zameer, Anila Usman), such as the time-varying signals generated by muscle activity. ANNs can 
automatically learn to recognize patterns, correlations, and relationships in the data, and use this 
knowledge to make accurate predictions or classifications.

In the case of EMG signal classification, ANNs can be trained to recognize different hand and 
finger movements based on the patterns of muscle activity captured by the EMG sensors. The ANNs 
can learn to extract relevant features from the raw EMG signals, and use these features to classify the 
movements into different categories, such as grasping, pointing, or pinching(W. Mo, Y. Huang, S. 
Zhang, E. Ip, D. C. Kilper, Y. Aono, and T. Tajima). ANNs can also be used to classify EMG signals in 
real-time, allowing for the control of prosthetic devices or human-computer interfaces in a seamless 
and natural way.

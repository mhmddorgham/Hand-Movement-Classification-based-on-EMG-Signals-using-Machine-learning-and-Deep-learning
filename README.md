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
gesture

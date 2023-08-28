# Anomaly-Detection-Autoencoder

The dataset contains satellite health monitoring data collected by the Indian Space Research Organisation (ISRO). This data provides insights into the status and parameters of a satellite, enabling the assessment of its overall health and performance. The dataset comprises observations recorded over a period of time, with a focus on various analog and digital parameters.

The data consists of the following columns:
1. Year
2. Day of year
3. Hour
4. Minute
5. Second
6. Millisecond
7. Microsecond
8. DTG-1_PCH_ANALOG_RAT (Analog)
9. DTG-1_ROL_ANALOG_RAT (Analog)
10. DTG-1_PCH_FINE_RATE (Analog)
11. DTG-1_ROL_FINE_RATE (Analog)
12. DTG-2_YAW_ANALOG_RAT (Analog)
13. DTG-2_PCH_ANALOG_RAT (Analog)
14. DTG-2_YAW_FINE_RATE (Analog)
15. DTG-2_PCH_FINE_RATE (Analog)
16. DTG-1_TH_TEMP (Analog)
17. DTG-1_ELECTRONICS_TH (Analog)
18. DTG-2_TH_TEMP (Analog)
19. DTG-2_ELECTRONICS_TH (Analog)
20. DTG-2_SYNC_STS (Digital)
21. DTG-2_ON_STS (Digital)
22. DTG-1_ON_STS (Digital)
23. DTG-2_RB_SUPPLY_STS (Digital)
24. DTG-1_SYNC_STS (Digital)
25. DTG-3_USBL_STS (Digital)
26. DTG_ANALOG_RATE_CHK (Digital)
27. DTG-2_TEMP_SEL_CMD_S (Digital)
28. DTG-1_RB_SUPPLY_STS (Digital)

This implementation combines data preprocessing techniques with an autoencoder neural network for anomaly detection. The key steps involve cleaning and transforming the input data, designing an autoencoder architecture, training the model, and detecting anomalies based on reconstruction errors.

Data Preprocessing

The provided data is read from a text file and transformed into a pandas DataFrame. Various preprocessing techniques are applied to prepare the data for the anomaly detection process:

Data Parsing: The data is read from the text file and separated using whitespace as a delimiter. The first 22 lines of the file are skipped, and the remaining rows are used to create the DataFrame.

Data Conversion:  The data includes date and time information which is initially read as strings and then converted to numeric values and then parsed and combined to create a datetime column. This column is moved to the beginning of the DataFrame for clarity.

Label Encoding: Columns with digital values are encoded using label encoding. This transformation converts categorical digital values into numerical labels, making them suitable for machine learning  and deep learning algorithms.

Feature Scaling: Numeric features are scaled using Min-Max scaling to ensure that all features have a similar scale, which helps in training the autoencoder more effectively.

The data now has the following columns: 

Data columns (total 22 columns):
No.   Column                          Non-Null Count   Dtype  
---  ------                          --------------   -----  
 0   Datetime                        186298 non-null  object 
 1   DTG-1_PCH_ANALOG_RAT (Analog)   186298 non-null  float64
 2   DTG-1_ROL_ANALOG_RAT (Analog)   186298 non-null  float64
 3   DTG-1_PCH_FINE_RATE (Analog)    186298 non-null  float64
 4   DTG-1_ROL_FINE_RATE (Analog)    186298 non-null  float64
 5   DTG-2_YAW_ANALOG_RAT (Analog)   186298 non-null  float64
 6   DTG-2_PCH_ANALOG_RAT (Analog)   186298 non-null  float64
 7   DTG-2_YAW_FINE_RATE (Analog)    186298 non-null  float64
 8   DTG-2_PCH_FINE_RATE (Analog)    186298 non-null  float64
 9   DTG-1_TH_TEMP (Analog)          186298 non-null  float64
 10  DTG-1_ELECTRONICS_TH (Analog)   186298 non-null  float64
 11  DTG-2_TH_TEMP (Analog)          186298 non-null  float64
 12  DTG-2_ELECTRONICS_TH (Analog)   186298 non-null  float64
 13  DTG-2_SYNC_STS (Digital)        186298 non-null  int64  
 14  DTG-2_ON_STS (Digital)          186298 non-null  int64  
 15  DTG-1_ON_STS (Digital)          186298 non-null  int64  
 16  DTG-2_RB_SUPPLY_STS (Digital)   186298 non-null  int64  
 17  DTG-1_SYNC_STS (Digital)        186298 non-null  int64  
 18  DTG-3_USBL_STS (Digital)        186298 non-null  int64  
 19  DTG_ANALOG_RATE_CHK (Digital)   186298 non-null  int64  
 20  DTG-2_TEMP_SEL_CMD_S (Digital)  186298 non-null  int64  
 21  DTG-1_RB_SUPPLY_STS (Digital)   186298 non-null  int64  

Anomaly Detection using Autoencoder

An autoencoder is an unsupervised machine learning technique that can learn efficient representations of input data by encoding it into a lower-dimensional space and then decoding it back to its original dimension. Anomalies can be detected by comparing the original data with the reconstructed data, using the reconstruction error as an indicator.

Model Architecture

The implemented model consists of an autoencoder neural network, which includes an encoder and a decoder. The encoder reduces the dimensionality of the input data, while the decoder attempts to reconstruct the original data from the encoded representation.

Encoder
The encoder is composed of three fully connected dense layers.
The input shape of the encoder is determined by the number of features in the input data.
Each dense layer has 64 units and uses the ReLU activation function, which introduces non-linearity to the network.
The final layer of the encoder reduces the dimensionality to the specified encoding dimension (5) using the ReLU activation function.
Decoder
The decoder is also composed of three fully connected dense layers.
The input shape of the decoder matches the encoding dimension.
Like the encoder, each dense layer of the decoder has 64 units and uses the ReLU activation function.
The output layer of the decoder reconstructs the data back to its original dimension using the ReLU activation function. 
Autoencoder
The autoencoder is constructed by stacking the encoder and decoder together.

Training and Evaluation
The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer. The autoencoder is trained to minimize the difference between the input data and the reconstructed output data. The training is performed for a specified number of epochs (10) with a batch size of 32.

After training, the model's performance is evaluated by calculating the reconstruction error for each input sample. The reconstruction error is the mean squared difference between the original input and the corresponding reconstructed output. Higher reconstruction errors are indicative of data points that deviate significantly from the learned patterns and are likely anomalies.

Anomaly Detection
A threshold is set for the reconstruction error to determine anomalies. The threshold is chosen based on the distribution of reconstruction errors in the training data. Data points with reconstruction errors above the threshold are considered anomalies. The index of the first anomaly is found using np.argmax and the corresponding datetime is obtained from the original datetime series.

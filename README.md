# Anomaly Detection Autoencoder

## Introduction

This project focuses on the task of anomaly detection using an autoencoder neural network. The dataset used contains satellite health monitoring data collected by the Indian Space Research Organisation (ISRO). This data provides valuable insights into the status and parameters of satellites, enabling the assessment of their overall health and performance. The dataset comprises observations recorded over time, with a focus on various analog and digital parameters.

### Dataset Columns

The dataset consists of the following columns:

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

## Data Preprocessing

### Data Parsing

The provided data is read from a text file and transformed into a pandas DataFrame. The initial rows are skipped, and the remaining data is parsed.

### Data Conversion

Date and time information in the dataset is initially read as strings and then converted to numeric values. These values are parsed and combined to create a datetime column, which is moved to the beginning of the DataFrame for clarity.

### Label Encoding

Columns with digital values are encoded using label encoding. This transformation converts categorical digital values into numerical labels, making them suitable for machine learning and deep learning algorithms.

### Feature Scaling

Numeric features are scaled using Min-Max scaling to ensure that all features have a similar scale, which helps in training the autoencoder more effectively.

## Anomaly Detection using Autoencoder

An autoencoder is employed for unsupervised anomaly detection. This technique learns efficient representations of input data by encoding it into a lower-dimensional space and then decoding it back to its original dimension. Anomalies are detected by comparing the original data with the reconstructed data, using the reconstruction error as an indicator.

### Model Architecture

The implemented model consists of an autoencoder neural network, which includes an encoder and a decoder.

#### Encoder
- The encoder consists of three fully connected dense layers.
- Each dense layer has 64 units and uses the ReLU activation function.
- The final layer of the encoder reduces the dimensionality to the specified encoding dimension (5) using the ReLU activation function.

#### Decoder
- The decoder also consists of three fully connected dense layers.
- Like the encoder, each dense layer of the decoder has 64 units and uses the ReLU activation function.
- The output layer of the decoder reconstructs the data back to its original dimension using the ReLU activation function.

#### Autoencoder
- The autoencoder is constructed by stacking the encoder and decoder together.

### Training and Evaluation

The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer. It is trained to minimize the difference between the input data and the reconstructed output data. Training is performed for a specified number of epochs (10) with a batch size of 32.

After training, the model's performance is evaluated by calculating the reconstruction error for each input sample. The reconstruction error is the mean squared difference between the original input and the corresponding reconstructed output. Higher reconstruction errors are indicative of data points that deviate significantly from the learned patterns and are likely anomalies.

### Anomaly Detection

Anomaly detection is performed by setting a threshold for the reconstruction error. This threshold is chosen based on the distribution of reconstruction errors in the training data. Data points with reconstruction errors above the threshold are considered anomalies. The index of the first anomaly is found using `np.argmax`, and the corresponding datetime is obtained from the original datetime series.

# Anomaly-Detection-Autoencoder

The dataset contains satellite health monitoring data collected by the Indian Space Research Organisation (ISRO). This data provides insights into the status and parameters of a satellite, enabling the assessment of its overall health and performance. The dataset comprises observations recorded over a period of time, with a focus on various analog and digital parameters.

This implementation combines data preprocessing techniques with an autoencoder neural network for anomaly detection. The key steps involve cleaning and transforming the input data, designing an autoencoder architecture, training the model, and detecting anomalies based on reconstruction errors.

Data Preprocessing

The provided data is read from a text file and transformed into a pandas DataFrame. Various preprocessing techniques are applied to prepare the data for the anomaly detection process:

Data Parsing: The data is read from the text file and separated using whitespace as a delimiter. The first 22 lines of the file are skipped, and the remaining rows are used to create the DataFrame.

Data Conversion:  The data includes date and time information which is initially read as strings and then converted to numeric values and then parsed and combined to create a datetime column. This column is moved to the beginning of the DataFrame for clarity.

Label Encoding: Columns with digital values are encoded using label encoding. This transformation converts categorical digital values into numerical labels, making them suitable for machine learning  and deep learning algorithms.

Feature Scaling: Numeric features are scaled using Min-Max scaling to ensure that all features have a similar scale, which helps in training the autoencoder more effectively.

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

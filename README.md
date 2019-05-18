This model is build with Keras API.
Building architecture:
	
	INPUT
	  |
	  v
   Bidirectional LSTM
          |
	  v
    Global MaxPooling
	  |
	  v
        Dense
	  |
          v
     Output Layer

Loss = Binary Crossentropy 
optimizer = Adam
lerning Rate = 0.05 (Choosed after experimentation)


import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

from .activation_mapping import activation_mapping
from .padding_mapping import padding_mapping

class CNNIndividual:
    def __init__(
            self, num_conv_layers, num_filters, filter_size, activation,
            pooling_size, dropout_prob, stride, padding, learning_rate,
            beta_1, beta_2,
        ):
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = activation
        self.pooling_size = pooling_size
        self.dropout_prob = dropout_prob
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.encode_step = 0
        self.fitness = None
        self.model = None

    def __repr__(self):
        return f"CNNIndividual(num_conv_layers={self.num_conv_layers}, num_filters={self.num_filters}, " \
               f"filter_size={self.filter_size}, activation={activation_mapping[self.activation]}, pooling_size={self.pooling_size}, " \
               f"dropout_prob={self.dropout_prob}, stride={self.stride}, padding={padding_mapping[self.padding]}, learning_rate={self.learning_rate}, " \
               f"beta_1={self.beta_1}, beta_2={self.beta_2})"
    
    
    def construct_model(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):
        model = Sequential()
        model.add(Input(shape=x_train.shape[1:]))

        for _ in range(self.num_conv_layers):
            model.add(Conv2D(self.num_filters, self.filter_size, activation=activation_mapping[self.activation], strides=self.stride, padding=padding_mapping[self.padding]))
            model.add(MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size)))
            if self.dropout_prob > 0:
                model.add(Dropout(self.dropout_prob))

        model.add(Flatten())
        model.add(Dense(y_train.shape[1], activation='softmax'))

        optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
        self.model = model

        return model
    
    def evaluate_fitness(self, x_val, y_val):
        _, accuracy = self.model.evaluate(x_val, y_val, verbose=0)

        self.fitness = accuracy

        return accuracy
    
    def encode(self, length=None):  
        values = [
            self.num_conv_layers,
            self.num_filters,
            self.filter_size - 1,
            self.activation,
            self.pooling_size - 1,
            int(self.dropout_prob * 10),
            self.stride,
            self.padding,
            int(self.learning_rate * 1000),
            int(self.beta_1 * 1000),
            int(self.beta_2 * 1000),
        ]
        
        if length is None:
            max_len = max(len(hex(value)[2:]) for value in values)
        else:
            max_len = length
        
        self.encode_step = max_len
        encoded_str = ''.join(hex(value)[2:].zfill(max_len) for value in values)
        return encoded_str

    @classmethod
    def decode(cls, encoded: str, step: int = 2):
        if len(encoded) % step != 0:
            raise ValueError("Invalid length of the encoded string")

        num_conv_layers = int(encoded[:step], 16)
        num_filters = int(encoded[step:2*step], 16)
        filter_size = int(encoded[2*step:3*step], 16) + 1
        activation = int(encoded[3*step:4*step], 16)
        pooling_size = int(encoded[4*step:5*step], 16) + 1
        dropout_prob = int(encoded[5*step:6*step], 16) / 10.0
        stride = int(encoded[6*step:7*step], 16)
        padding = int(encoded[7*step:8*step], 16)
        learning_rate = int(encoded[8*step:9*step], 16) / 1000.0
        beta_1 = int(encoded[9*step:10*step], 16) / 1000.0
        beta_2 = int(encoded[10*step:11*step], 16) / 1000.0

        return cls(num_conv_layers, num_filters, filter_size, activation, pooling_size, dropout_prob, stride, padding, learning_rate, beta_1, beta_2)
    
    # uniform mutation
    def mutate(self, mutation_rate=0.05): 
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError("Mutation rate must be between 0 and 1")

        if random.random() < mutation_rate:
            self.num_conv_layers = random.randint(1, 2)

        if random.random() < mutation_rate:
            self.num_filters = random.randint(16, 64)

        if random.random() < mutation_rate:
            self.filter_size = random.choice([3, 5])

        if random.random() < mutation_rate:
            self.activation = random.randint(0, len(activation_mapping) - 1)

        if random.random() < mutation_rate:
            self.pooling_size = random.choice([2, 3])

        if random.random() < mutation_rate:
            dropout = random.uniform(-1, 1)
            if dropout > 0:
                self.dropout_prob = dropout
            else:
                self.dropout_prob = 0

        if random.random() < mutation_rate:
            self.stride = random.choice([1, 2])

        if random.random() < mutation_rate:
            self.padding = random.randint(0, len(padding_mapping) - 1)

        if random.random() < mutation_rate:
            self.learning_rate = 10 ** (-random.uniform(1, 5))

        if random.random() < mutation_rate:
            self.beta_1 = random.uniform(0.8, 0.999)

        if random.random() < mutation_rate:
            self.beta_2 = 1 - 10 ** (-random.uniform(1, 5))


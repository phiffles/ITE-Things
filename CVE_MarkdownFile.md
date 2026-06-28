# Part B Researchers Playground.
 In Part B, I applied advanced computer vision techniques I had not learned before, beyond the baseline CNN, to improve my model accuracy, reduce overfitting and to learn new computer vision techniques.

## Techniques Chosen:
 I went to experiment on Batch Normalization, Data Augmentation, Ensembling, Transfer Learning Using MobileNetV2. 

## Technique One: Batch Normalization

### 1. The Concept
 Batch Normalization, a deep learning technique used to improve neural network(process of repeatedly feeding data into a model and adjusting its parameters so it can make a more accurate prediction over time) training. Batch Normalization works by normalizing the outputs of a layer before passing them to the next layer. I used it so that it can improve my model's training stability and model performance.

### 2. Implementation
 I implemented it by adding a Batch Normalization layer after each convolutional and dense layer in my CNN model.

 Code:
```
from keras.layers import BatchNormalization   # Import the BatchNormalization layer

# Add a convolutional layer with 32 filters, each of size 3x3
# 'relu' activation introduces non-linearity, 'same' padding keeps output size equal to input size
improved_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))

# Add Batch Normalization to stabilize and speed up training
improved_model.add(BatchNormalization())

# Add a MaxPooling layer with a 2x2 window
# This downsamples the feature maps by taking the maximum value in each 2x2 block
improved_model.add(MaxPooling2D((2,2)))
```

### 3. The Learning
 Batch Normalization works by keeping the scale of activation consistent in each layer so that it training will be smoother and more reliable. This would reduce a problem known as internal covariate shift, where the distribution of inputs to each layer keeps changing during training. By stabilizing these distributions, the model can learn faster and more reliably.

 It also provides a form of regularization effect, by adding small variations during training to help to reduce overfitting and improving generalization.(focus on understanding the concepts, so you can answer new questions you haven’t seen before)

 In my experiment, my baseline CNN achieved an accuracy of approximately 0.79. After adding Batch Normalization, my improved model achieved 0.89 accuracy. Which shows that Batch normalization improved the overall performance of my model.

## Technique Two: Data Augmentation

### 1. The Concept
 For my second technique, I explored on Data Augmentation. Data augmentation is used to artificially increase the size and diversity of the training dataset by applying random transformation to the images.

### 2. Implementation
 I implemented Data Augmentation using the ImageDataGenerator from Keras.

 Code:
```
# Create an ImageDataGenerator for real-time data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,        # Randomly rotate images up to 10 degrees
    width_shift_range=0.05,   # Randomly shift images horizontally by up to 5% of width
    height_shift_range=0.05,  # Randomly shift images vertically by up to 5% of height
    zoom_range=0.05           # Randomly zoom in/out by up to 5%
)

# Train the augmented model using the generator
history_aug = improved_model_aug.fit(
    datagen.flow(X_train, y_train, batch_size=32),  # Generate augmented batches of training data
    validation_data=(X_val, y_val),                 # Use validation set for monitoring performance
    epochs=15                                       # Train for 15 epochs
)
```

### 3. The learning
 Data augmentation works by generating slightly modified versions of the original images. For example rotating the images, move the image up down left or right  and zoom in or out. There are more Data augmentation methods like brightness adjustment which I did not try as well as colour shifting which slightly changes the RGB Values.This would expose the model to more variations of the data, which would improve its ability to generalize and recognize images more precisely and accurately instead of memorizing the training data. After using Data Augmentation, my model's accuracy increased from 0.79 to 0.86.
 
## Technique Three: Ensembling

### 1. The Concept
 For my third technique. I started using Ensemble on my notebook. Ensemble is an technique used to combine multiple group models together to produce a more reliable prediction. Instead of relying on a single model where there could be biasness. Ensemble averages predictions , reduce variance and improves generalization.

### 2. Implementation
 I implemented Ensembling by averaging the predictions of the two different models trained on different techniques. This would combine the probability outputs of each model making the final classification decision.

 Code:
```
# Generate predictions on the test set using the first improved model
prediction_1 = improved_model.predict(X_test)

# Generate predictions on the test set using the augmented improved model
prediction_2 = improved_model_aug.predict(X_test)

# Average the predictions from both models to form the ensemble prediction
ensemble_pred = (prediction_1 + prediction_2) / 2

# Convert the averaged probabilities into class labels by selecting the index of the highest probability
ensemble_classes = np.argmax(ensemble_pred, axis=1)

# Print the accuracy score of the ensemble compared to the true labels
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_classes))

# Print a detailed classification report (precision, recall, f1-score, support) for the ensemble predictions
print(classification_report(y_test, ensemble_classes))
```

### 3. The learning
 Ensemble works by combining multiple predictions from multiple models instead of using only one model. I learnt that different CNN models can learn diferent features from the card images and learnt that a model might recognize shapes better while another may focus more on colours and patterns. After using the Ensemble Technique I achieved an accuracy of 0.90.

## Technique Four: Transfer Learning Using MobileNetV2

### 1. The Concept
 For my fourth technique I explored on Transfer Learning (MobileNetV2). What is transfer learning? it is a technique where a model that was pretrained on a large data set is reused as the starting point of a new task. But instead of training a neural network from scratch, i borrowed the feature extraction layers that learnt how to recognise general patterns like edges shapes and textures from millions of images. This would save training time because the dataset that i have is small.

### 2. Implementation
 I implemented Transfer Learning by loading MobileNetV2 pre-trained on ImageNet , whilst freezing its base layer so that the what the model learned so far would not be overwritten and adding a new classification head ontop to predict my specific classes.

 Code:
```
# Load the MobileNetV2 model with pre-trained ImageNet weights, excluding the top classification layers
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

# Freeze the base model so its weights are not updated during training
base_model.trainable = False

# Apply Global Average Pooling to reduce the feature maps into a single vector per image
x = GlobalAveragePooling2D()(base_model.output)

# Add a fully connected (dense) layer with 128 neurons and ReLU activation
x = Dense(128, activation="relu")(x)

# Add Dropout with a rate of 0.3 to reduce overfitting by randomly disabling 30% of neurons during training
x = Dropout(0.3)(x)

# Final output layer with 'num_classes' neurons and softmax activation for multi-class classification
outputs = Dense(num_classes, activation="softmax")(x)

# Define the full transfer learning model by connecting the base model input to the new output layer
transfer_model = Model(base_model.input, outputs)
```

### 3. The learning
 Transfer learning works by reusing the feature extraction layers from a model trained from big datasets and only trains a new classifier on top. and i learnt that even though MobileNetV2 was not trained on my specific images, the low level features such as edges and texttures were useful for my classificaition. However, i only achieved an accuracy of 0.60 which was way lower than i expected and i asked the ai why and learnt that it was due to three reasons. Firstly my dataset was very small and spread across 20 over classes which is not enough data for the model to learn reliable patterns. Secondly, I used an image size of 128x128 when mobilenet was originally designed for 224x224 which means the features were less detailed. Third I only trained the new classification head without any tuning of my base model layers hence the model couldnt adapt to its features of my specific dataset.

## Technique Five: Random Erasing (Cutout) Data Augmentation

### 1. The Concept
 For my fourth technique I decided to explore abit more on Data Augmentation. On Random Erasing , Cutout Data Augmentation. Random erasing is a technique where a random section of the image is erased during training. And the purpose is to force the model to focus on multiple parts of the image instead of just relying heavily on one specific feature. This would help my model improve its ability to generalise and become much more robust when onjects are partially blocked or hidden. Since my data set was small random erasing helped increase image variation without collecting more data.

### 2. Implementation
 I implemented this technique by creating a custom keras layer that randomlly removes a small rectangular patch from my training images. The layer is then only applied during training so that validation and testing images would remain unchanged. I then added this custom layer at the start of my CNN model before the convolutional layers.

Code Snippet:
```
# Custom Random Erasing layer
class RandomErasing(Layer):
    def __init__(self, probability=0.5, area_fraction=0.1, **kwargs):
        super(RandomErasing, self).__init__(**kwargs)
        self.probability = probability
        self.area_fraction = area_fraction

    def call(self, inputs, training=None):

        # Only apply augmentation during training
        if not training:
            return inputs

        def apply_erasing(img):
            h = tf.shape(img)[0]
            w = tf.shape(img)[1]

            # Define the size of the erased patch
            erase_h = tf.cast(tf.cast(h, tf.float32) * self.area_fraction, tf.int32)
            erase_w = tf.cast(tf.cast(w, tf.float32) * self.area_fraction, tf.int32)

            # Random starting position
            top = tf.random.uniform([], 0, h - erase_h, dtype=tf.int32)
            left = tf.random.uniform([], 0, w - erase_w, dtype=tf.int32)

            # Create the mask
            y_grid = tf.range(h)[:, tf.newaxis]
            x_grid = tf.range(w)[tf.newaxis, :]

            mask = ~((y_grid >= top) & (y_grid < top + erase_h) &
                     (x_grid >= left) & (x_grid < left + erase_w))

            mask = tf.cast(mask[:, :, tf.newaxis], img.dtype)

            return img * mask

        return tf.map_fn(
            lambda img: tf.cond(
                tf.random.uniform([]) < self.probability,
                lambda: apply_erasing(img),
                lambda: img
            ),
            inputs
        )

# Build the CNN model
cutout_model = Sequential()

# Add Random Erasing layer
cutout_model.add(RandomErasing(probability=0.5, area_fraction=0.1))

# CNN feature extraction layers
cutout_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
cutout_model.add(BatchNormalization())
cutout_model.add(MaxPooling2D((2, 2)))

cutout_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cutout_model.add(BatchNormalization())
cutout_model.add(MaxPooling2D((2, 2)))

# Fully connected layers
cutout_model.add(Flatten())
cutout_model.add(Dense(128, activation='relu'))
cutout_model.add(BatchNormalization())
cutout_model.add(Dense(53, activation='softmax'))
```

### 3. The learning
 Random Erasing works by randomly hiding parts of training images so that the model would learn to recognise objects from multiple features instead of memorising only one important area. Through this technique i learnt that data augmentation can improve the robustness of a model without needing additional training data. I also learnt that applying augmentation only during training is important because validation and test images should remain unchanged for fair evaluation. However, Random Erasing also has limitations because if the erased region is too large the model may lose important information and struggle to learn correctly. In my implementation i used a small erase area of 10% with a probability of 0.5 so that the augmentation would improve generalisation without overly damaging the images.

## Overall Learning.
 Out of all the techniques I experimented with, Batch Normalization gave me the best results. In my convolutional neural network, the convolutional layers extracted features at increasing levels of complexity, then the flattening and dense layers combined those features into decision making. And by normalizing the outputs at each stage, Batch Normalization provided training stability and reduced overfitting, which led to better generalization. As a result, it delivered the highest accuracy among all the methods I tested.(shows that it performs not only on my training set but it performs well on validation and test data.)
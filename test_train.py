import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
import numpy as np

# Mocking the model building
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False, weights="imagenet")(input_layer)
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(512, activation="relu")(f1)
    class1 = Dropout(0.5)(class1)
    class2 = Dense(1, activation="sigmoid")(class1)
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(512, activation="relu")(f2)
    regress1 = Dropout(0.5)(regress1)
    regress2 = Dense(4, activation="sigmoid")(regress1)
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

class FaceTracker(Model): 
    def __init__(self, model, **kwargs): 
        super().__init__(**kwargs)
        self.model = model

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.optimizer = opt

    def train_step(self, batch):
        X, y = batch
        
        with tf.GradientTape() as tape:            
            classes, coords = self.model(X, training=True)
            
            # Ensure shapes are known to avoid the ValueError
            classes = tf.ensure_shape(classes, [None, 1])
            coords = tf.ensure_shape(coords, [None, 4])
            y_class = tf.ensure_shape(y[0], [None, 1])
            y_coord = tf.ensure_shape(y[1], [None, 4])
            
            batch_classloss = self.closs(y_class, classes)
            mask = tf.cast(y_class, tf.float32)
            batch_localizationloss = self.lloss(tf.cast(y_coord, tf.float32) * mask, coords * mask)
            
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            
        grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
    
    def test_step(self, batch): 
        X, y = batch
        classes, coords = self.model(X, training=False)
        
        classes = tf.ensure_shape(classes, [None, 1])
        coords = tf.ensure_shape(coords, [None, 4])
        y_class = tf.ensure_shape(y[0], [None, 1])
        y_coord = tf.ensure_shape(y[1], [None, 4])
        
        batch_classloss = self.closs(y_class, classes)
        mask = tf.cast(y_class, tf.float32)
        batch_localizationloss = self.lloss(tf.cast(y_coord, tf.float32) * mask, coords * mask)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

# Mock data
X_sample = np.random.rand(8, 120, 120, 3).astype(np.float32)
y_class = np.random.randint(0, 2, (8, 1)).astype(np.uint8)
y_box = np.random.rand(8, 4).astype(np.float32)

facetracker = build_model()
model = FaceTracker(facetracker)
model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanSquaredError())

# Single step test
results = model.train_step((X_sample, (y_class, y_box)))
print("Train step results:", results)

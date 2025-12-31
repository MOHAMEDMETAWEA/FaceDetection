import tensorflow as tf
import numpy as np

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

# Mock data for batch size 8
y_true = np.random.rand(8, 4).astype(np.float32)
yhat = np.random.rand(8, 4).astype(np.float32)

loss = localization_loss(y_true, yhat)
print(f"Loss with batch size 8: {loss.numpy()}")

# Mock data for batch size 1
y_true_1 = y_true[:1]
yhat_1 = yhat[:1]
loss_1 = localization_loss(y_true_1, yhat_1)
print(f"Loss with batch size 1: {loss_1.numpy()}")

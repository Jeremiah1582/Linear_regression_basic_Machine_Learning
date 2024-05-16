import numpy as np 
from sklearn.linear_model import LinearRegression

# TRAINING MODEL
def training_model():
    x = np.array([[1], [2], [3],[4],[5],[6],[7],[8],[9],[10]])#training data
    y = np.array([2,4,6,8,10,12,14,16,18,20])
    
    model = LinearRegression()
    model.fit(x,y)
    
    return model

trained_model = training_model()

# PREDICTION MODEL
def make_prediction(model, value):
    return model.predict(np.array([[value]]))

print(make_prediction(trained_model, 15)) #output 30
from features import *

todaysinput = np.random.randn(10, 425)
targets = np.random.randn(10)

trained_weights = trainingloop(
    epochs=100,
    hiddensize=64,
    inputsize=425,
    todaysinput=todaysinput,
    targets=targets,
    learningrate=0.001
)
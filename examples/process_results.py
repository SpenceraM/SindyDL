import pandas as pd
import numpy as np
import pickle


if __name__ == '__main__':
    with open('C:/Users/smhoc/OneDrive/Documents/projects/SindyDL/examples/lorenz/experiment_results_202307120841.pkl', 'rb') as f:
        data = pickle.load(f)

    print(data)
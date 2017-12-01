import pandas as pd
import numpy as np
import re
import model

def paramPipeline(steps, model):
    
    params = []
    for key, value in model.parameters.items():
            temp = [key,value]
            params.append(temp)
    
    pipe_parameters = []
    for step in steps:
        step_dict = {}   
        extra = step + "__"
        
        for param in params:
            if(extra in param[0]):
                step_dict[re.sub(extra, '',param[0])] = param[1]
            
        pipe_parameters.append(step_dict)
    
    return pipe_parameters

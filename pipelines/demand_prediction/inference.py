"""
TODO This seems to be very sagemaker specific, lets return here once we figured out what to do
when we work on the productionalization.

"""
import json
import numpy as np

def input_handler(data, context):
    data_str = data.read().decode('utf-8')
    lines = data_str.split('\n')
    transformer_instances = []
    for js in lines:
        obj = json.loads(js)
        raw = np.array(obj['flat_values'], dtype=np.float32)
        n   = obj['n']
        d   = obj['d']
        transformed_instances.append(raw.reshape((n, d)))
    return transformewd

def output_handler(response, context):
    pass



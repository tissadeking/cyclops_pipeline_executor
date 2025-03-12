import pickle
from datetime import datetime

def save_model_fun(model, model_name):
    # Get current time as string
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # save the model to disk
    filename = model_name + ' ' + time_str + ' .sav'
    pickle.dump(model, open(filename, 'wb'))
    # Serialise the file
    #with open(filename, 'wb') as handle:
    #    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


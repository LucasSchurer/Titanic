import os
from datetime import datetime

def get_available_path(directory, filename, extension) :
    full_path = os.path.join(directory, '{}.{}'.format(filename, extension))
    
    if not os.path.exists(directory) :
        os.makedirs(directory)
        
    if os.path.exists(full_path) :
        i = 0
        new_file = '{}{}.{}'.format(filename, i, extension)
        while os.path.exists(os.path.join(directory, new_file)) :
            i += 1
            new_file = '{}{}.{}'.format(filename, i, extension)
        
        full_path = os.path.join(directory, new_file)

    return full_path
    
def get_current_time() :
    time = datetime.now()
    time = time.strftime('%Y%m%d%H%M%S')
    return time
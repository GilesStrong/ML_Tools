from sklearn.model_selection import train_test_split

def uncert_round(value, uncert):
    if uncert == 0:
        return value, uncert
    
    factor = 1.0
    while uncert/factor > 1:
        factor *= 10.0
    
    value /= factor
    uncert /= factor
        
    i = 0    
    while uncert*(10**i) <= 1:
        i += 1
    
    roundUncert = factor*round(uncert, i)
    roundValue = factor*round(value, i)
    if int(roundUncert) == roundUncert:
        roundUncert = int(roundUncert)
        roundValue = int(roundValue)
    return roundValue, roundUncert

def split_dev_val(inData, size=0.2, seed=1337):
    return train_test_split([i for i in inData.index.tolist()], test_size=size, random_state=seed)
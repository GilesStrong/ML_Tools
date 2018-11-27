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
    
    round_uncert = factor*round(uncert, i)
    round_value = factor*round(value, i)
    if int(round_uncert) == round_uncert:
        round_uncert = int(round_uncert)
        round_value = int(round_value)
    return round_value, round_uncert

def split_dev_val(inData, size=0.2, seed=1337):
    return train_test_split([i for i in inData.index.tolist()], test_size=size, random_state=seed)
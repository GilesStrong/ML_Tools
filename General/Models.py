from __future__ import division

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

def getModel(version, nIn, compileArgs, mode, nOut=1):
    model = Sequential()

    if 'depth' in compileArgs:
        depth = compileArgs['depth']
    else:
        depth = 3
    if 'width' in compileArgs:
        width = compileArgs['width']
    else:
        width = 100
    if 'do' in compileArgs:
        do = compileArgs['do']
    else:
        do = False
    if 'bn' in compileArgs:
        bn = compileArgs['bn']
    else:
        bn = False

    if "modelRelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal'))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('relu'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal'))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('relu'))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)

    elif "modelSelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='VarianceScaling'))
        model.add(Activation('selu'))
        if do: model.add(AlphaDropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='VarianceScaling'))
            model.add(Activation('selu'))
            if do: model.add(AlphaDropout(do))

    elif "modelSwish" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal'))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('swish'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal'))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('swish'))
            if bn == 'post': model.add(BatchNormalization())
            if do: Dropout(do)
    
    if 'class' in mode:        
        if nOut == 1:
            model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        else:
            model.add(Dense(nOut, activation='softmax', kernel_initializer='glorot_normal'))

    elif 'regress' in mode:
        model.add(Dense(nOut, activation='linear', kernel_initializer='glorot_normal'))

    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        optimiser = Adam(lr=compileArgs['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        
    model.compile(loss=compileArgs['loss'], optimizer=optimiser)
    return model
from __future__ import division

from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization,  Concatenate, Embedding, Input, Reshape
from keras.optimizers import *
from keras.regularizers import *
from keras.models import Sequential

'''
Todo:
- Combine get_model methods
- Works out way to remove need for dense layer for continuous inputs in cat model
'''

def get_model(version, nIn, compileArgs, mode, nOut=1):
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
    if 'l2' in compileArgs:
        reg = l2(compileArgs['l2'])
    else:
        reg = None

    if "modelRelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('relu'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('relu'))
            if bn == 'post': model.add(BatchNormalization())
            if do: model.add(Dropout(do))

    elif "modelSelu" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='lecun_normal', kernel_regularizer=reg))
        model.add(Activation('selu'))
        if do: model.add(AlphaDropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg))
            model.add(Activation('selu'))
            if do: model.add(AlphaDropout(do))

    elif "modelSwish" in version:
        model.add(Dense(width, input_dim=nIn, kernel_initializer='he_normal', kernel_regularizer=reg))
        if bn == 'pre': model.add(BatchNormalization())
        model.add(Activation('swish'))
        if bn == 'post': model.add(BatchNormalization())
        if do: model.add(Dropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg))
            if bn == 'pre': model.add(BatchNormalization())
            model.add(Activation('swish'))
            if bn == 'post': model.add(BatchNormalization())
            if do: model.add(Dropout(do))
    
    if 'class' in mode:        
        if nOut == 1:
            model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        else:
            model.add(Dense(nOut, activation='softmax', kernel_initializer='glorot_normal'))

    elif 'regress' in mode:
        model.add(Dense(nOut, activation='linear', kernel_initializer='glorot_normal'))

    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        if 'amsgrad' not in compileArgs: compileArgs['amsgrad'] = False
        if 'beta_1' not in compileArgs: compileArgs['beta_1'] = 0.9
        optimiser = Adam(lr=compileArgs['lr'], beta_1=compileArgs['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compileArgs['amsgrad'])

    if compileArgs['optimizer'] == 'sgd':
        if 'momentum' not in compileArgs: compileArgs['momentum'] = 0.9
        if 'nesterov' not in compileArgs: compileArgs['nesterov'] = False
        optimiser = SGD(lr=compileArgs['lr'], momentum=compileArgs['momentum'], decay=0.0, nesterov=compileArgs['nesterov'])
        
    model.compile(loss=compileArgs['loss'], optimizer=optimiser)
    return model

def get_cat_model(version, nContIn, compileArgs, mode, nOut=1, cat_szs=[]):
    #Categorical embeddings
    models = []
    for cat_sz in cat_szs:
        model = Sequential()
        embedding_size = min((cat_sz+1)//2, 50)
        model.add(Embedding(cat_sz, embedding_size, input_length=1))
        model.add(Reshape(target_shape=(embedding_size,)))
        models.append(model)
    
    #Continuous inputs
    if nContIn:
        model = Sequential()
        model.add(Dense(nContIn, input_dim=nContIn, kernel_initializer='glorot_normal'))
        models.append(model)
    
    merged = Concatenate()([x.output for x in models])

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
    if 'l2' in compileArgs:
        reg = l2(compileArgs['l2'])
    else:
        reg = None

    if "modelRelu" in version:
        merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
        if bn == 'pre': merged = BatchNormalization()(merged)
        merged = Activation('relu')(merged)
        if bn == 'post': merged = (BatchNormalization())(merged)
        if do: merged = Dropout(do)(merged)
        for i in range(depth):
            merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
            if bn == 'pre': merged = BatchNormalization()(merged)
            merged = Activation('relu')(merged)
            if bn == 'post': merged = BatchNormalization()(merged)
            if do: merged = Dropout(do)(merged)

    elif "modelSelu" in version:
        merged = Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg)(merged)
        merged = Activation('selu')(merged)
        if do: merged = AlphaDropout(do)(merged)
        for i in range(depth):
            merged = Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg)(merged)
            merged = Activation('selu')(merged)
            if do: merged = AlphaDropout(do)(merged)

    elif "modelSwish" in version:
        merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
        if bn == 'pre': merged = BatchNormalization()(merged)
        merged = Activation('swish')(merged)
        if bn == 'post': merged = BatchNormalization()(merged)
        if do: merged = Dropout(do)(merged)
        for i in range(depth):
            merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
            if bn == 'pre': merged = BatchNormalization()(merged)
            merged = Activation('swish')(merged)
            if bn == 'post': merged = BatchNormalization()(merged)
            if do: merged = Dropout(do)(merged)
    
    if 'class' in mode:        
        if nOut == 1:
            merged = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(merged)
        else:
            merged = Dense(nOut, activation='softmax', kernel_initializer='glorot_normal')(merged)

    elif 'regress' in mode:
        merged = Dense(nOut, activation='linear', kernel_initializer='glorot_normal')(merged)
        
    model = Model([x.input for x in models], merged)

    if 'lr' not in compileArgs: compileArgs['lr'] = 0.001
    if compileArgs['optimizer'] == 'adam':
        if 'amsgrad' not in compileArgs: compileArgs['amsgrad'] = False
        if 'beta_1' not in compileArgs: compileArgs['beta_1'] = 0.9
        optimiser = Adam(lr=compileArgs['lr'], beta_1=compileArgs['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compileArgs['amsgrad'])

    if compileArgs['optimizer'] == 'sgd':
        if 'momentum' not in compileArgs: compileArgs['momentum'] = 0.9
        if 'nesterov' not in compileArgs: compileArgs['nesterov'] = False
        optimiser = SGD(lr=compileArgs['lr'], momentum=compileArgs['momentum'], decay=0.0, nesterov=compileArgs['nesterov'])
    
    
    model.compile(loss=compileArgs['loss'], optimizer=optimiser)
    return model
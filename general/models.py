from __future__ import division

from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization,  Concatenate, Embedding, Input, Reshape
from keras.optimizers import *
from keras.regularizers import *
from keras.models import Sequential

'''
Todo:
- Combine getM_model methods
- Works out way to remove need for dense layer for continuous inputs in cat model
'''

def get_model(version, n_in, compile_args, mode, n_out=1):
    model = Sequential()

    if 'depth' in compile_args:
        depth = compile_args['depth']
    else:
        depth = 3
    if 'width' in compile_args:
        width = compile_args['width']
    else:
        width = 100
    if 'do' in compile_args:
        do = compile_args['do']
    else:
        do = False
    if 'bn' in compile_args:
        bn = compile_args['bn']
    else:
        bn = False
    if 'l2' in compile_args:
        reg = l2(compile_args['l2'])
    else:
        reg = None

    if "modelRelu" in version:
        model.add(Dense(width, input_dim=n_in, kernel_initializer='he_normal', kernel_regularizer=reg))
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
        model.add(Dense(width, input_dim=n_in, kernel_initializer='lecun_normal', kernel_regularizer=reg))
        model.add(Activation('selu'))
        if do: model.add(AlphaDropout(do))
        for i in range(depth):
            model.add(Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg))
            model.add(Activation('selu'))
            if do: model.add(AlphaDropout(do))

    elif "modelSwish" in version:
        model.add(Dense(width, input_dim=n_in, kernel_initializer='he_normal', kernel_regularizer=reg))
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
        if n_out == 1:
            model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
        else:
            model.add(Dense(n_out, activation='softmax', kernel_initializer='glorot_normal'))

    elif 'regress' in mode:
        model.add(Dense(n_out, activation='linear', kernel_initializer='glorot_normal'))

    if 'lr' not in compile_args: compile_args['lr'] = 0.001
    if compile_args['optimizer'] == 'adam':
        if 'amsgrad' not in compile_args: compile_args['amsgrad'] = False
        if 'beta_1' not in compile_args: compile_args['beta_1'] = 0.9
        optimiser = Adam(lr=compile_args['lr'], beta_1=compile_args['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compile_args['amsgrad'])

    if compile_args['optimizer'] == 'sgd':
        if 'momentum' not in compile_args: compile_args['momentum'] = 0.9
        if 'nesterov' not in compile_args: compile_args['nesterov'] = False
        optimiser = SGD(lr=compile_args['lr'], momentum=compile_args['momentum'], decay=0.0, nesterov=compile_args['nesterov'])
        
    model.compile(loss=compile_args['loss'], optimizer=optimiser)
    return model

def get_cat_model(version, n_cont_n, compile_args, mode, n_out=1, cat_szs=[]):
    #Categorical embeddings
    models = []
    for cat_sz in cat_szs:
        model = Sequential()
        embedding_size = min((cat_sz+1)//2, 50)
        model.add(Embedding(cat_sz, embedding_size, input_length=1))
        model.add(Reshape(target_shape=(embedding_size,)))
        models.append(model)
    
    #Continuous inputs
    if n_cont_n:
        model = Sequential()
        model.add(Dense(n_cont_n, input_dim=n_cont_n, kernel_initializer='glorot_normal'))
        models.append(model)
    
    merged = Concatenate()([x.output for x in models])

    if 'depth' in compile_args:
        depth = compile_args['depth']
    else:
        depth = 3
    if 'width' in compile_args:
        width = compile_args['width']
    else:
        width = 100
    if 'do' in compile_args:
        do = compile_args['do']
    else:
        do = False
    if 'bn' in compile_args:
        bn = compile_args['bn']
    else:
        bn = False
    if 'l2' in compile_args:
        reg = l2(compile_args['l2'])
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
        if n_out == 1:
            merged = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(merged)
        else:
            merged = Dense(n_out, activation='softmax', kernel_initializer='glorot_normal')(merged)

    elif 'regress' in mode:
        merged = Dense(n_out, activation='linear', kernel_initializer='glorot_normal')(merged)
        
    model = Model([x.input for x in models], merged)

    if 'lr' not in compile_args: compile_args['lr'] = 0.001
    if compile_args['optimizer'] == 'adam':
        if 'amsgrad' not in compile_args: compile_args['amsgrad'] = False
        if 'beta_1' not in compile_args: compile_args['beta_1'] = 0.9
        optimiser = Adam(lr=compile_args['lr'], beta_1=compile_args['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compile_args['amsgrad'])

    if compile_args['optimizer'] == 'sgd':
        if 'momentum' not in compile_args: compile_args['momentum'] = 0.9
        if 'nesterov' not in compile_args: compile_args['nesterov'] = False
        optimiser = SGD(lr=compile_args['lr'], momentum=compile_args['momentum'], decay=0.0, nesterov=compile_args['nesterov'])
    
    
    model.compile(loss=compile_args['loss'], optimizer=optimiser)
    return model

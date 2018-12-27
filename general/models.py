from __future__ import division
from typing import Dict, Any, Union, Callable
from tensorflow.python.framework.ops import Tensor

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, AlphaDropout, Dropout, BatchNormalization, Concatenate, Embedding, Reshape, Input
from keras.optimizers import Optimizer, Adam, SGD
from keras.regularizers import l2

from .activations import swish, Swish

'''
Todo:
- Refactor cat model
- Add res & dense models
- Combine get_model methods
- Works out way to remove need for dense layer for continuous inputs in cat model
'''


class ModelBuilder(object):
    def __init__(self, objective:str, n_in:int, n_out:int,
                 model_args:Dict[str,Any]={}, opt_args:Dict[str,Any]={},
                 loss:Union[Callable[[float],float],str,'auto']='auto'):
        self.objective,self.n_in,self.n_out = objective.lower(),n_in,n_out
        self.parse_loss(loss)
        self.parse_model_args(model_args)
        self.parse_opt_args(opt_args)

    def parse_loss(self, loss:Union[Callable[[float],float],str,None]=None) -> None:
        if loss is 'auto':
            if 'class' in self.objective:
                self.loss = 'categorical_crossentropy' if self.n_out > 1 and 'multi' in self.objective else 'binary_crossentropy'
            else:
                self.loss = 'mean_squared_error'
        else:   
                self.loss = loss

    def parse_model_args(self, model_args:Dict[str,Any]) -> None:
        model_args = {k.lower(): model_args[k] for k in model_args}
        self.width = 100    if 'width' not in model_args else model_args['width']
        self.depth = 4      if 'depth' not in model_args else model_args['depth']
        self.do    = 0      if 'do'    not in model_args else model_args['do']
        self.bn    = False  if 'bn'    not in model_args else model_args['bn']
        self.l2    = False  if 'l2'    not in model_args else model_args['l2']
        self.act   = 'relu' if 'act'   not in model_args else model_args['act'].lower()
    
    def parse_opt_args(self, opt_args:Dict[str,Any]) -> None:
        opt_args = {k.lower(): opt_args[k] for k in opt_args}
        self.opt = 'adam' if 'opt' not in opt_args else opt_args['opt']
        if self.opt not in ['adam', 'sgd']: raise ValueError('Optimiser not currently available')
        self.opt_args = {k: opt_args[k] for k in opt_args if k != 'opt'}        

    def build_opt(self) -> Optimizer:
        if   self.opt == 'adam': return Adam(**self.opt_args)
        elif self.opt == 'sgd':  return SGD(**self.opt_args)

    def set_lr(self, lr:float) -> None:
        self.opt_args['lr'] = lr

    @staticmethod
    def lookup_init(act:str) -> str:
        if act == 'relu':    return 'he_normal'
        if act == 'selu':    return 'lecun_normal'
        if act == 'sigmoid': return 'glorot_normal'
        if act == 'softmax': return 'glorot_normal'
        if act == 'linear':  return 'glorot_normal'
        if 'swish' in act:   return 'he_normal'

    def layer_fn(self, x:Tensor) -> Tensor:
        reg = None if self.l2 is None else l2(self.l2)
        if 'swish' not in self.act:
            act = Activation(self.act)
        elif 'swish' in self.act and 'train' in self.act:
            act = Swish(trainable=True) 
        elif 'swish' in self.act and 'train' not in self.act:
            act = Swish(trainable=False) 

        x = Dense(self.width, kernel_initializer=self.lookup_init(self.act), kernel_regularizer=reg)(x)
        x = act(x)
        if self.bn: x = BatchNormalization()(x)
        if self.do: x = Dropout(self.do)(x)
        return x

    def get_head(self) -> Tensor:
        return Input(shape=(self.n_in,))

    def get_body(self, x:Tensor, depth:int) -> Tensor:
        for d in range(depth): x = self.layer_fn(x)
        return x

    def get_tail(self, x:Tensor) -> Tensor:
        if 'class' in self.objective:
            if 'multi' in self.objective: 
                tail = Dense(self.n_out, activation='softmax', kernel_initializer=self.lookup_init('softmax'))     
            else:
                tail = Dense(self.n_out, activation='sigmoid', kernel_initializer=self.lookup_init('sigmoid'))
        else:
                tail = Dense(self.n_out, activation='linear',  kernel_initializer=self.lookup_init('linear'))
        return tail(x)

    def build_model(self) -> Model:
        head = self.get_head()
        body = self.get_body(head, self.depth)
        tail = self.get_tail(body)
        return Model(inputs=head, outputs=tail)

    def get_model(self) -> Model:
        model = self.build_model()
        opt = self.build_opt()
        model.compile(optimizer=opt, loss=self.loss)
        return model


# def get_cat_model(version, n_cont_n, compile_args, mode, n_out=1, cat_szs=[]):
#     # Categorical embeddings
#     models = []
#     for cat_sz in cat_szs:
#         model = Sequential()
#         embedding_size = min((cat_sz + 1) // 2, 50)
#         model.add(Embedding(cat_sz, embedding_size, input_length=1))
#         model.add(Reshape(target_shape=(embedding_size,)))
#         models.append(model)
    
#     # Continuous inputs
#     if n_cont_n:
#         model = Sequential()
#         model.add(Dense(n_cont_n, input_dim=n_cont_n, kernel_initializer='glorot_normal'))
#         models.append(model)
    
#     merged = Concatenate()([x.output for x in models])

#     if 'depth' in compile_args:
#         depth = compile_args['depth']
#     else:
#         depth = 3
#     if 'width' in compile_args:
#         width = compile_args['width']
#     else:
#         width = 100
#     if 'do' in compile_args:
#         do = compile_args['do']
#     else:
#         do = False
#     if 'bn' in compile_args:
#         bn = compile_args['bn']
#     else:
#         bn = False
#     if 'l2' in compile_args:
#         reg = l2(compile_args['l2'])
#     else:
#         reg = None

#     if "modelRelu" in version:
#         merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
#         if bn == 'pre': merged = BatchNormalization()(merged)
#         merged = Activation('relu')(merged)
#         if bn == 'post': merged = (BatchNormalization())(merged)
#         if do: merged = Dropout(do)(merged)
#         for i in range(depth):
#             merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
#             if bn == 'pre': merged = BatchNormalization()(merged)
#             merged = Activation('relu')(merged)
#             if bn == 'post': merged = BatchNormalization()(merged)
#             if do: merged = Dropout(do)(merged)

#     elif "modelSelu" in version:
#         merged = Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg)(merged)
#         merged = Activation('selu')(merged)
#         if do: merged = AlphaDropout(do)(merged)
#         for i in range(depth):
#             merged = Dense(width, kernel_initializer='lecun_normal', kernel_regularizer=reg)(merged)
#             merged = Activation('selu')(merged)
#             if do: merged = AlphaDropout(do)(merged)

#     elif "modelSwish" in version:
#         merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
#         if bn == 'pre': merged = BatchNormalization()(merged)
#         merged = Activation(swish)(merged)
#         if bn == 'post': merged = BatchNormalization()(merged)
#         if do: merged = Dropout(do)(merged)
#         for i in range(depth):
#             merged = Dense(width, kernel_initializer='he_normal', kernel_regularizer=reg)(merged)
#             if bn == 'pre': merged = BatchNormalization()(merged)
#             merged = Activation(swish)(merged)
#             if bn == 'post': merged = BatchNormalization()(merged)
#             if do: merged = Dropout(do)(merged)
    
#     if 'class' in mode:        
#         if n_out == 1:
#             merged = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(merged)
#         else:
#             merged = Dense(n_out, activation='softmax', kernel_initializer='glorot_normal')(merged)

#     elif 'regress' in mode:
#         merged = Dense(n_out, activation='linear', kernel_initializer='glorot_normal')(merged)
        
#     model = Model([x.input for x in models], merged)

#     if 'lr' not in compile_args: compile_args['lr'] = 0.001
#     if compile_args['optimizer'] == 'adam':
#         if 'amsgrad' not in compile_args: compile_args['amsgrad'] = False
#         if 'beta_1' not in compile_args: compile_args['beta_1'] = 0.9
#         optimiser = Adam(lr=compile_args['lr'], beta_1=compile_args['beta_1'], beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=compile_args['amsgrad'])

#     if compile_args['optimizer'] == 'sgd':
#         if 'momentum' not in compile_args: compile_args['momentum'] = 0.9
#         if 'nesterov' not in compile_args: compile_args['nesterov'] = False
#         optimiser = SGD(lr=compile_args['lr'], momentum=compile_args['momentum'], decay=0.0, nesterov=compile_args['nesterov'])
    
#     model.compile(loss=compile_args['loss'], optimizer=optimiser)
#     return model

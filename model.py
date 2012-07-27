import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal.downsample import *
from theano.tensor.nnet.conv import *

import sys,os,pdb
import shutil


def find_last_epoch(path):
    files = os.listdir(path)

    allnum =[]

    for f in files:
        s = f.split('.cp')
        if len(s) == 1 or s[0].find('config')>=0 or s[0].find('current')>=0:
            continue
        allnum.append(int(s[0].split('_')[1]))
    if len(allnum)>0:
        return max(allnum)
    else:
        return 0



class logistic(object):
    def __init__(self,i_size,h_size, W = None, b = None, c = None):
        """                                                                                                                                                                             h_size : number of hidden units                                                                                                                                                 e_size : embedding size                                                                                                                                                         p_size : patch size (input size)                                                                                                                                                """

        self.h_size = h_size
        self.i_size = i_size
        wbound = numpy.sqrt(6./(self.i_size+self.h_size))
        if W == None:
            W = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound,size = (self.i_size, self.h_size)), dtype = theano.config.floatX)
        if b == None:
            b = numpy.asarray( numpy.zeros((self.h_size,)), dtype = theano.config.floatX)

        self.params = { 'W': theano.shared(value = W, name = 'cae_w'),
                        'b': theano.shared(value = b, name = 'cae_b')}

    def save(self,i,name=""):
        cPickle.dump( { 'W' : self.params['W'].get_value(),
                        'b' : self.params['b'].get_value()} , open('files/'+name+'logistic_'+str(i)+'.pkl','w'),protocol=-1)

    def get_jacobi(self,x,bs):
        h = self.get_hidden(x)
        W = self.params['W']
        a = T.reshape(h*(1-h),(bs,1,self.h_size))
        Jx = T.reshape(W,(1,self.i_size,self.h_size))
        return Jx*a

    def get_hidden(self,x):
        h = T.nnet.sigmoid(T.dot(x,self.params['W']) + self.params['b'])
        return h


class cae(object):
    def __init__(self,i_size,h_size, W = None, b = None, c = None):
        """
        h_size : number of hidden units
        e_size : embedding size
        p_size : patch size (input size)
        """
        self.h_size = h_size
        self.i_size = i_size
        wbound = numpy.sqrt(6./(self.i_size+self.h_size))
        if W == None:
            W = numpy.asarray( numpy.random.uniform( low = -wbound, high = wbound,size = (self.i_size, self.h_size)), dtype = theano.config.floatX)
        if b == None:
            b = numpy.asarray( numpy.zeros((self.h_size,)), dtype = theano.config.floatX)
        if c == None:
            c = numpy.asarray( numpy.zeros((self.i_size,)), dtype = theano.config.floatX)

        self.params = { 'W': theano.shared(value = W, name = 'cae_w'),
                        'b': theano.shared(value = b, name = 'cae_b'),
                        'c': theano.shared(value = c, name = 'cae_c') }


    def save(self,i,name=""):
        cPickle.dump( { 'W' : self.params['W'].get_value(),
                        'b' : self.params['b'].get_value(),
                        'c' : self.params['c'].get_value() } , open('files/'+name+'cae_'+str(i)+'.pkl','w'),protocol=-1)

    def get_jacobi(self,x,bs):
        if bs == 1:
            h = self.get_hidden(x).flatten()
            return self.params['W']* (h*(1-h))
        else:
            h = self.get_hidden(x)
            W = self.params['W']
            a = T.reshape(h*(1-h),(bs,1,self.h_size))
            Jx = T.reshape(W,(1,self.i_size,self.h_size))
            return Jx*a

    def get_reconstruction_jacobi(self,x):
        r = self.get_reconstruction(x)
        f = r.flatten()
        W = self.params['W'].T
        return (f * W)

    def get_stocha_jacobi(self,x,nx):
        h = self.get_hidden(x)
        nh = self.get_hidden(nx)
        return mse(h,nh)

    def get_hidden(self,x):
        h = T.nnet.sigmoid(T.dot(x,self.params['W']) + self.params['b'])
        return h


    def get_reconstruction(self,x):
        h = self.get_hidden(x)
        return T.nnet.sigmoid(T.dot(h,self.params['W'].T) + self.params['c'])

    def get_linear_reconstruction(self,x):
        h = self.get_hidden(x)
        return T.dot(h,self.params['W'].T) + self.params['c']

    def get_linear_jacobi(self,x,bs):
        h = self.get_hidden(x)
        W = self.params['W']
        a = T.reshape(h*(1-h),(bs,1,self.h_size))
        Jx = T.reshape(W**2,(1,self.i_size,self.h_size))
        return Jx*a



def ce(x,y):
    return T.mean(-T.sum((x*T.log(y) + (1.-x)*T.log(1.-y)), axis=1))

def e(x,y):
    return T.mean(-T.sum((x*T.log(y)),axis=1))
def L2(x):
    return T.sum(x**2)

def mse(x,y):
    return T.mean(T.sum((x-y)**2,axis=1))

def L1(x):
    return T.sum(T.abs_(x))

def nll(o,y):
    return -T.mean(T.log(o)[T.arange(y.shape[0]),y]+T.sum(T.log(1-o), axis=1)-T.log(1-o)[T.arange(y.shape[0]),y])

def nlls(o,y):
    return -T.mean( (T.log(o)[T.arange(y.shape[0]),y]+T.sum(T.log(1-o), axis=1)-T.log(1-o)[T.arange(y.shape[0]),y]) * T.neq(y,-1))


def shuffle_array(d1,d2):
    x1,y1 = d1
    x2,y2 = d2
    y1 = numpy.array(numpy.zeros(y1.shape) -1,dtype='int32')
    x = numpy.vstack((x1,x2))
    y = numpy.hstack((y1,y2))
    idx = range(len(x))
    numpy.random.shuffle(idx)
    print 'Shuffling data', idx[:10]
    return x[idx],y[idx]


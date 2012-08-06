#!/usr/bin/env python

from jobscheduler import JobmanInterface as JB
from tools.hyperparams import hyperparams
from model import *
import cPickle
import theano
import theano.tensor as T
from theano.tensor.signal.downsample import *
from theano.tensor.nnet.conv import *
from dataset.DatasetInterfaces import root
from dataset.base import create_files

import sys,os,pdb
import shutil
import misc as mi
from svm_linear import eval_fold

class experiment(object):
    def __init__(self,jobman):
        self.jobman = jobman
        self.x = T.matrix()
        self.y = T.imatrix()

        self.data = root(self.jobman.state['dataset'],prepro_fct_x=[],prepro_fct_y=[])

        self.shared_x,self.shared_y = self.data.get_train_labeled()

        self.n_i,self.n_h = (self.data.get_sample_size(),self.jobman.state['n_h'])

        self.featsplit = int(self.n_h * self.jobman.state['featsplit'])

        self.cae = cae(self.n_i,self.n_h)
        self.logreg = logistic(self.featsplit,self.jobman.state['nclass'])

    def getcosts(self):
        bs = self.jobman.state['bs']

        # Costs to penalize the regular CAE term and
        # L1 penalty on the weights for regularisation

        self.costs_str = ['disc_contraction','nuisance_contraction','disc_sparsity','nuisance_sparsity','cda','discrimination','reconstruction']

        self.J = self.cae.get_jacobi(self.x,1)
        self.J_d, self.J_o = (self.J[:,:self.featsplit],self.J[:,self.featsplit:])

        self.nll = nlls(self.logreg.get_hidden(self.cae.get_hidden(self.x)[:,:self.featsplit]),self.y.flatten())

        self.reconstruction = self.cae.get_reconstruction(self.x)
        if self.jobman.state['rectype'] == 'ce':
            self.recons = ce(self.x, self.reconstruction)
        else:
            self.recons = mse(self.x, self.reconstruction)

        self.costs = [ self.jobman.state['lambdad']*L2(self.J_d),
                       self.jobman.state['lambdao']*L2(self.J_o),
                       self.jobman.state['l1d']*L1(self.cae.params['W'][:,:self.featsplit]),
                       self.jobman.state['l1o']*L1(self.cae.params['W'][:,self.featsplit:]),
                       self.jobman.state['ortho']*L2(T.dot(self.J_d.T,self.J_o)),
                       self.jobman.state['disc']*self.nll,
                       self.recons ]


        return numpy.sum( self.costs )

    def run(self):

        print '--- Loading data'
        self.jobman.save()
        if not os.path.exists('files'):
            os.mkdir('files')
        bs = self.jobman.state['bs']

        params = self.cae.params.values() + self.logreg.params.values()

        cost = self.getcosts()

        # Computing gradient of the sum of all costs
        # on all the parameters of the model 
        # (except the convolutional part)

        updates = dict ( (p,p - self.jobman.state['lr']*g) for p,g in zip(params,T.grad(cost,params)) )
        
        k = T.iscalar()

        # Compiling optimisation theano function

        optimize = theano.function([k],self.costs,updates=updates,givens = { self.x:self.shared_x[k:k+1],
                                                                             self.y:self.shared_y[k:k+1] })
        self.jobman.state['valid_discriminant'] = 0
        self.jobman.state['valid_nuisance'] = 0 
        self.jobman.state['valid_both'] = 0
        print '--- Training'
        sys.stdout.flush()
        bs = self.jobman.state['bs']
        costs = []

        h = self.cae.get_hidden(self.x)
        h_d = h[:,:self.featsplit]
        h_o = h[:,self.featsplit:]

        for i in range(self.jobman.state['epochs']):
            costs = []

            # For each epoch we take 50% of labeled data
            # and 50% of unlabeled data shuffled in random
            # order.

            self.data.get_train_labeled(i)

            # We evaluate the representations using an SVM
            # with 'evalfreq' frequency.
            
            if (i+1)%self.jobman.state['evalfreq'] == 0:
                self.svm(h_d,'discriminant')
                self.svm(h_o,'nuisance')
                self.svm(h,'both')
                self.cae.save(i)
                self.logreg.save(i,'logreg')
                mi.view_data(self.cae.params['W'].get_value(borrow=True).T,'files/cae_'+str(i))
                self.jobman.save()

            total = self.shared_x.get_value(borrow=True).shape[0]/bs

            # We loop over the samples of the current split
            # to tune the parameters of the model.

            for j in range(total):
                costs.append(optimize(j))
                if j%self.jobman.state['freq'] == 0:
                    #self.evalreg()
                    print i, j,zip(self.costs_str,numpy.mean(numpy.array(costs),axis=0))
                    sys.stdout.flush()
            self.jobman.state['cft']=i
        self.svm(h_d,'discriminant')
        self.svm(h_o,'nuisance')
        self.svm(h,'both')


    def evalreg(self,name=""):
        # Function to evaluate the classification error
        # when using the output of the logistic regression.
        # This function is used only for monitoring purposes

        print '--- Loading data'
        self.jobman.save()

        o = self.logrege.get_hidden(self.cae_emotions.get_hidden(self.x))

        f = theano.function([self.x,self.y],hardmax(o,self.y))

        data = root(self.jobman.state['dataset'],prepro_fct_x=[],prepro_fct_y=[])

        X,Y = data.get_train_labeled(shared_array=False)
        print '--- Logreg Valid Accuracy:',f(X,Y)

        return True

    def svm(self,h,name=""):

        # Function that trains an SVM on the representation
        # h given in argument.

        print '--- Loading data'
        self.jobman.save()

        f = theano.function([self.x],[h])
        def fix_theano(X,l):
            return f(X)[0]


        self.svm_data = root(self.jobman.state['dataset'],prepro_fct_x=[(fix_theano,[])])

        res = eval_fold(3,self.svm_data)
        cPickle.dump(res,open('results.pkl','w'))
        self.jobman.state['train_'+name] = res['train']
        if self.jobman.state['valid_'+name] <= res['valid']:
            self.jobman.state['valid_'+name] = res['valid']
            self.jobman.state['test_'+name] = res['test']
        self.jobman.save()
        print name,res
        return True


def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    exp = experiment(jobhandler)
    exp.run()
    return 0

if __name__ == "__main__":
    HP_init = [ ('values','dataset',['/scratch/rifaisal/data/mnistds/mnistvanilla_files_dict.pkl']),
                ('values','evalfreq',[1]),
                ('values','freq',[2000]),
                ('values','model',[None]),
                ('values','nclass',[10]),
                # Model hyper-parameters below
                ('values','rectype',['ce']),
                ('uniform','lr',[.0005,.002]),
                ('values','ortho',[0.,3.,5.,7.,10.]),
                ('values','disc',[.1]),
                ('values','bs',[1]),
                ('values','epochs',[600]),
                ('values','n_h',[500]),
                ('values','featsplit',[.5]),
                ('values','l1d',[.00001,.000001]),
                ('values','l1o',[.00001,.000001]),
                ('uniform','lambdad',[.005,.01]),
                ('uniform','lambdao',[.005,.01])]

    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)

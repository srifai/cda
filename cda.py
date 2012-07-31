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
        self.y = T.ivector()

        def fix_label(Y,l):
            return numpy.array(Y-1,dtype='int32').flatten()

        self.data = root(self.jobman.state['dataset'],prepro_fct_x=[],prepro_fct_y=[(fix_label,[])])

        dataset_list = ['/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_fold0_files_dict.pkl',
                        '/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_fold1_files_dict.pkl',
                        '/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_fold2_files_dict.pkl',
                        '/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_fold3_files_dict.pkl',
                        '/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_fold4_files_dict.pkl']


        csplit = self.jobman.state['split']

        assert csplit < 5

        self.data.fusion(dataset_list[csplit:csplit+1])

        x,y = shuffle_array(self.data.get_train_unlabeled(shared_array=False),self.data.get_train_labeled(shared_array=False))

        self.shared_x,self.shared_y = (theano.shared(value=x),theano.shared(value=y))
        

        self.n_i,self.n_hi,self.n_he = (self.data.get_sample_size(),self.jobman.state['n_hi'],self.jobman.state['n_he'])
        

        self.cae_emotions = cae(self.n_i,self.n_he)
        self.cae_identity = cae(self.n_i,self.n_hi)
        
        self.logrege = logistic(self.n_he,7)

        if self.jobman.state['identity']:
            self.logregi = logistic(self.n_hi,900)


    def init_costs(self):
        bs = self.jobman.state['bs']


        # Costs to penalize the regular CAE term and
        # L1 penalty on the weights for regularisation
        # on both identity and emotion encoders

        self.cae_emotions_costs_list = ['emot jacobi','emot L1']
        self.cae_emotions_costs = [ self.jobman.state['dhdxe']*L2(self.cae_emotions.get_jacobi(self.x,bs))/bs,
                                    self.jobman.state['l1e']*L1(self.cae_emotions.params['W']) ]


        self.cae_identity_costs_list = ['identity jacobi','identity L1']
        self.cae_identity_costs = [ self.jobman.state['dhdxe']*L2(self.cae_identity.get_jacobi(self.x,bs))/bs,
                                    self.jobman.state['l1i']*L1(self.cae_identity.params['W']) ]
                                          


        # Supervised cost to backpropagate labels into
        # the emotion encoder.

        self.logrege_costs_list = ['logreg']
        self.logrege_costs = [ nlls(self.logrege.get_hidden(self.cae_emotions.get_hidden(self.x)),self.y) ]


        if self.jobman.state['identity']:
            self.logregi_costs = [ nll(self.logregi.get_hidden(self.cae_identity.get_hidden(self.x)),self.y[1]) ]
        else:
            self.logregi_costs = [0]

        # Orthoganility term between the jacobians of 
        # both encoders.

        self.orthogonality = T.dot(self.cae_emotions.get_jacobi(self.x,1).T,self.cae_identity.get_jacobi(self.x,1))

        # Total reconstruction cost taken as the sum
        # of the reconstruction of each AE.

        self.reconstruction = T.nnet.sigmoid(self.cae_emotions.get_linear_reconstruction(self.x) + self.cae_identity.get_linear_reconstruction(self.x))

        self.mixed_costs_list = ['reconstruction','orthogonality']
        self.mixed_costs  = [  ce(self.x, self.reconstruction),self.jobman.state['ortho']*L2(self.orthogonality) ]


        self.total_cost_str = self.cae_emotions_costs_list + self.cae_identity_costs_list + self.logrege_costs_list + self.mixed_costs_list

        self.total_cost_list = self.cae_emotions_costs + self.cae_identity_costs + self.logrege_costs + self.mixed_costs

        self.total_cost = numpy.sum( self.total_cost_list )

        return self.total_cost


    def run(self):

        print '--- Loading data'
        self.jobman.save()
        if not os.path.exists('files'):
            os.mkdir('files')
        bs = self.jobman.state['bs']

        params = self.cae_emotions.params.values() + self.cae_identity.params.values() + self.logrege.params.values()

        self.cost = self.init_costs()

        # Computing gradient of the sum of all costs
        # on all the parameters of the model 
        # (except the convolutional part)

        self.updates = dict ( (p,p - self.jobman.state['lr']*g) for p,g in zip(params,T.grad(self.cost,params)) )
        
        k = T.iscalar()

        # Compiling optimisation theano function

        self.optimize = theano.function([k],self.total_cost_list,updates=self.updates,givens = { self.x:self.shared_x[k:k+1],
                                                                                                 self.y:self.shared_y[k:k+1] })
        self.jobman.state['valid_emotion'] = 0
        self.jobman.state['valid_identity'] = 0 
        self.jobman.state['valid_both'] = 0
        print '--- Training'
        sys.stdout.flush()
        bs = self.jobman.state['bs']
        costs = []
        #self.evalreg()
        for i in range(self.jobman.state['epochs']):
            costs = []

            # For each epoch we take 50% of labeled data
            # and 50% of unlabeled data shuffled in random
            # order.

            x,y = shuffle_array(self.data.get_train_unlabeled(i,shared_array=False),self.data.get_train_labeled(i,shared_array=False))
            self.shared_x.set_value(x)
            self.shared_y.set_value(y)

            # We evaluate the representations using an SVM
            # with 'evalfreq' frequency.
            
            if (i+1)%self.jobman.state['evalfreq'] == 0:
                he = self.cae_emotions.get_hidden(self.x)
                hi = self.cae_identity.get_hidden(self.x)
                self.svm(he,'emotion')
                self.svm(hi,'identity')
                self.svm(T.concatenate([he,hi],axis=1),'both')
                self.cae_emotions.save(i,'emot')
                self.cae_identity.save(i,'iden')
                self.logrege.save(i,'logreg')
                mi.view_data(self.cae_emotions.params['W'].get_value(borrow=True).T,'files/emot_'+str(i))
                mi.view_data(self.cae_identity.params['W'].get_value(borrow=True).T,'files/iden_'+str(i))
                self.jobman.save()


            total = self.shared_x.get_value(borrow=True).shape[0]/bs

            # We loop over the samples of the current split
            # to tune the parameters of the model.

            for j in range(total):
                costs.append(self.optimize(j))
                if j%self.jobman.state['freq'] == 0:
                    #self.evalreg()
                    print i, j,zip(self.total_cost_str,numpy.mean(numpy.array(costs),axis=0))
                    sys.stdout.flush()



            self.jobman.state['cft']=i
        self.svm()


    # Function to evaluate the classification error
    # when using the output of the logistic regression

    def evalreg(self,name=""):
        # This function is used only for monitoring purposes

        print '--- Loading data'
        self.jobman.save()
        o = self.logrege.get_hidden(self.cae_emotions.get_hidden(self.x))

        f = theano.function([self.x,self.y],hardmax(o,self.y))

        def fix_label(Y,l):
            return numpy.array(Y-1,dtype='int32').flatten()

        data = root(self.jobman.state['dataset'],prepro_fct_x=[],prepro_fct_y=[(fix_label,[])])

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


    def get_conv(self,x):
        a = conv2d(x,
                   self.W,
                   filter_shape = self.filter,
                   image_shape = self.image,
                   unroll_kern = 4,
                   unroll_batch = 4,
                   border_mode = str(self.jobman.state['border']))
        return T.nnet.sigmoid(a + self.b.dimshuffle('x',0,'x','x')) 

    def meanpool(self,x):
        pool = self.jobman.state['pool'][0]
        pool_size = (self.image[-1]-self.filter[-1]+1)/pool
        outputs = []
        for i in range(pool):
            for j in range(pool):
                x_start = i*pool_size
                x_fin = (i+1)*pool_size
                y_start = j*pool_size
                y_fin = (j+1)*pool_size
                outputs.append(x[:,:,x_start:x_fin,y_start:y_fin].mean(axis=[2,3]))

        return T.concatenate(outputs,axis=1)
       
    def maxpool(self,x):
        return max_pool_2d(x, self.jobman.state['pool'], ignore_border=True)


def jobman_entrypoint(state, channel):
    jobhandler = JB.JobHandler(state,channel)
    exp = experiment(jobhandler)
    exp.run()
    return 0


if __name__ == "__main__":
    HP_init = [ ('values','dataset',['/scratch/rifaisal/data/tfdconv/convfeat_patch14_1024_33_files_dict.pkl']),
                ('values','evalfreq',[5]),
                ('values','split',[3]),
                ('values','model',[None]),
                ('values','identity',[0]),
                ('uniform','lr',[.0005,.002]),
                ('values','ortho',[0.,3.,5.,7.,10.]),
                ('values','bs',[1]),
                ('values','epochs',[600]),
                ('values','n_hi',[1000]),
                ('values','n_he',[1000]),
                ('values','freq',[2000]),
                ('values','l1e',[.00001,.000001]),
                ('values','l1i',[.00001,.000001]),
                ('uniform','dhdxe',[.005,.01]),
                ('uniform','dhdxi',[.005,.01])]
    hp_sampler = hyperparams(HP_init)
    jobparser = JB.JobParser(jobman_entrypoint,hp_sampler)

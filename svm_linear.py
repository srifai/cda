import sys
import cPickle
import gzip
import time

import theano
import numpy
import scipy.io
import sklearn.svm
import sklearn.linear_model


from dataset.DatasetInterfaces import root
from os.path import join

import pdb

def compute_score(model, x, y):
    ''' returns the accuracy in per cent of the model'''
    acc = (model.predict(x) == y).sum()
    return float(acc)*100. / y.shape[0]

def OptimizeLinearSVM(datasets, MAXSTEPS=10, verbose=1):
    ''' performs a line search of nsteps of the C
    regularization parameter of the linear svm'''
    

    Cs = [2**-i for i in numpy.linspace(-25,25,281)]
    # Cs = [2*(-15), 2*(-14.75),...,2*(19.75),2*(20)]
    step_idx=16
    current_idx=150
    new_idx=current_idx+step_idx
    Ccurrent = Cs[current_idx]
    Cnew = Cs[new_idx]
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    C_to_allstats = {}
    Cbest = None
    
    if verbose: print >> sys.stderr, "Performing line search to get the best C (%d steps)"% MAXSTEPS 
    steps=1
    while steps < MAXSTEPS:
        steps+=1
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            model = scikits.learn.svm.LinearSVC(C=Ccurrent)
            model.fit(train_set_x, train_set_y)
            train = compute_score(model, train_set_x, train_set_y)
            valid = compute_score(model, valid_set_x, valid_set_y)
            test = compute_score(model, test_set_x, test_set_y)

            C_to_allstats[Ccurrent] = {'train_acc':train,\
                'valid_acc':valid, 'test_acc':test} 

        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            model = scikits.learn.svm.LinearSVC(C=Cnew)
            model.fit(train_set_x, train_set_y)
            train = compute_score(model, train_set_x, train_set_y)
            valid = compute_score(model, valid_set_x, valid_set_y)
            test = compute_score(model, test_set_x, test_set_y)

            C_to_allstats[Cnew] = {'train_acc':train,\
                'valid_acc':valid, 'test_acc':test} 

        # If Cnew has a higher val acc than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew]['valid_acc'] > C_to_allstats[Ccurrent]['valid_acc']:
            if verbose: 
                print >> sys.stderr, \
                    "\tval accuracy[Cnew %f] = %f > val accuracy[Ccurrent %f] = %f" % \
                    (Cnew, C_to_allstats[Cnew]['valid_acc'], Ccurrent, C_to_allstats[Ccurrent]['valid_acc'])
            if Cbest is None or C_to_allstats[Cnew]['valid_acc'] > C_to_allstats[Cbest]['valid_acc']:
                Cbest = Cnew
                if verbose: 
                    print >> sys.stderr, \
                    "\tNEW BEST: Cbest <= %f, val accuracy[Cbest] = %f" %\
                    (Cbest, C_to_allstats[Cbest]['valid_acc'])
            Ccurrent = Cnew
            current_idx = new_idx
            new_idx = current_idx+step_idx
            Ccurrent = Cs[current_idx]
            Cnew = Cs[new_idx]
            if verbose: 
                print >> sys.stderr,\
                "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" %\
                (step_idx, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            if verbose: 
                print >> sys.stderr,\
                "\tval accuracy[Cnew %f] = %f < val accuracy[Ccurrent %f] = %f" %\
                (Cnew, C_to_allstats[Cnew]['valid_acc'], Ccurrent, C_to_allstats[Ccurrent]['valid_acc'])
            if Cbest is None or C_to_allstats[Ccurrent]['valid_acc'] > C_to_allstats[Cbest]['valid_acc']:
                Cbest = Ccurrent
                if verbose: 
                    print >> sys.stderr,\
                    "\tCbest <= %f, val accuracy[Cbest] = %f"%\
                    (Cbest, C_to_allstats[Cbest]['valid_acc'])
            step_idx = numpy.asarray(step_idx/1.25 +1,dtype=numpy.int)
            new_idx = current_idx-step_idx
            Cnew = Cs[new_idx] #current * Cstepfactor
            if verbose: 
                print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (step_idx, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    if verbose:
        for C in allC:
            print >> sys.stderr,\
            "\ttrain val test accuracy[C %f] = [%f , %f , %f]" %\
            (C, C_to_allstats[C]['train_acc'],C_to_allstats[C]['valid_acc'],C_to_allstats[C]['test_acc']),
            if C == Cbest: print >> sys.stderr, " *best*"
            else: print >> sys.stderr, ""
    else:
        print >> sys.stderr, "\tBestC %f with Validation Accuracy = %f" % (Cbest, C_to_allstats[Cbest])

    return Cbest, C_to_allstats

def eval_fold(nsteps, dataset):
    '''
    performs a log-linear search of nsteps to find the best C
    of a linear SVM on the dataset 'name'
    
    returns a dictionnary where:

    'C' is the best regularization param wrt valid accuracy
    'train' is the train accuracy for C
    'valid' is the valid accuracy for C
    'test' is the test accuracy for C
    'Optimize' is the dictionnary storing the whole line search process
    '''

    print "Loading data..."

    train_set_x,train_set_y = dataset.get_all_train_labeled(shared_array=False)
    valid_set_x,valid_set_y = dataset.get_all_valid(shared_array=False)
    test_set_x,test_set_y = dataset.get_all_test(shared_array=False)
   
    train_set_y = train_set_y.flatten()
    valid_set_y = valid_set_y.flatten()
    test_set_y = test_set_y.flatten()

    bestC, res = OptimizeLinearSVM(([train_set_x, train_set_y],\
                        [valid_set_x, valid_set_y],\
                        [test_set_x, test_set_y]),\
                        MAXSTEPS=nsteps)    
    
    best = {'C':bestC,'train':res[bestC]['train_acc'],
            'valid':res[bestC]['valid_acc'],
            'test':res[bestC]['test_acc'],
            'Optimize':res}

    return best


if __name__ == '__main__':
    data = root('/scratch/rifaisal/data/paradata/msrparaphrasefullconvl2_files_dict.pkl',prepro_fct_x=[(scale,[])])
    print 'Starting evaluation...'
    eval_fold(nsteps=30, dataset = data)
    print dic

from cPickle import load
from copy import copy
from numpy import random,floor,array,sum,concatenate
from theano import config,shared
from warnings import warn
import os
import shutil
import pdb
names = ['train_unlabeled','train_labeled','valid','test']

class root(object):

    def __init__(self,dictpath,prepro_fct_x=[],prepro_fct_y=[],debug=False):

        def strip_dict_path(dictpath):
            dic = load(open(dictpath))
            abs_path = os.path.split(dictpath)[0]
            return dict( (k,[os.path.join(abs_path,os.path.split(f)[1]) for f in v]) for k,v in dic.iteritems())

        self._default_dict = strip_dict_path(dictpath)
        self._current_dict = copy(self._default_dict)
        self._prepro_x = prepro_fct_x
        self._prepro_y = prepro_fct_y
        self._debug = debug



    def add_prepro_x(self,tup):
        self._prepro_x += [tup]

    def add_prepro_y(self,tup):
        self._prepro_y += [tup]


    def fusion(self,datasetpaths,axis=0):
        ''' 
        Fusion of a list of datasets into one dataset
        '''

        def strip_dict_path(dictpath):
            dic = load(open(dictpath))
            abs_path = os.path.split(dictpath)[0]
            return dict( (k,[os.path.join(abs_path,os.path.split(f)[1]) for f in v]) for k,v in dic.iteritems())



        if not axis:
            for dictpath in datasetpaths:
                _temp_dict = strip_dict_path(dictpath)
                for k,v in _temp_dict.iteritems():
                    self._default_dict[k] += v

            self._current_dict = copy(self._default_dict)
        else:
            raise Exception('Not implemented')

    def copy_files(self,new_path):
        '''
        this will copy all the files in the dict to another directory
        and modify the l_default_dict and _current_dict to reflect the new
        path. This can be used to make a local copy of the dataset on the local
        machine instead of the network

        THIS function must be called just after the initialisation of the dicts, before
        setting the unlabeld trained etc.
        '''
        new_dict={}

        for key in self._current_dict.keys():
            new_dict[key]=[]
            for elem in self._current_dict[key]:
                path,name=os.path.split(elem)
                old_full_name=elem
                new_full_name=os.path.join(new_path,name)
                shutil.copy(old_full_name,new_full_name)
                new_dict[key].append(new_full_name)

        self._original_dict=copy(self._current_dict)
        self._default_dict=copy(new_dict)
        self._current_dict=new_dict


    def modify_file_paths(self,new_path):
        '''
        this will modify the loaded copy of the dictionnary so it
        can be used on any cluster without having to re-create the dict
        this is temporary solution unitl we freeze the methodology 
        for working on different lcusters

        THIS function must be called just after the initialisation of the dicts, before
        setting the unlabeld trained etc.
        '''
        new_dict={}
        for key in self._current_dict.keys():
            new_dict[key]=[]
            for elem in self._current_dict[key]:
                path,name=os.path.split(elem)
                old_full_name=elem
                new_full_name=os.path.join(new_path,name)
                new_dict[key].append(new_full_name)

        self._original_dict=copy(self._current_dict)
        self._default_dict=copy(new_dict)
        self._current_dict=new_dict
        

    def __get_unlabeled_train_split(self,i):
        return load(open(self._current_dict[names[0]][i]))

    def __get_train_split(self,i):
        return load(open(self._current_dict[names[1]][i]))

    def __get_valid_split(self,i):
        return load(open(self._current_dict[names[2]][i]))

    def __get_test_split(self,i):
        return load(open(self._current_dict[names[3]][i]))

    def __open_file(self,path):
        _tuple = load(open(path))
        if len(self._prepro_x):
            _ctuplex = _tuple[0]
            for _prepro_fct,_prepro_param in self._prepro_x:
                _ctuplex = _prepro_fct(_ctuplex,_prepro_param)
            _tuplex = _ctuplex
        else:
            _tuplex = _tuple[0]
        if len(self._prepro_y):
            _ctupley = _tuple[1]
            for _prepro_fct,_prepro_param in self._prepro_y:
                _ctupley = _prepro_fct(_ctupley,_prepro_param)
            _tupley = _ctupley
        else:
            _tupley = _tuple[1]

        return (_tuplex,_tupley)

    def set_train_unlabeled(self,include_train=True,include_valid=True,include_test=True):
        if include_train:
            self._current_dict[names[0]]+= self._default_dict[names[1]]
            print 'Merged train with unlabeled train'
        if include_valid:
            self._current_dict[names[0]]+= self._default_dict[names[2]]
            print 'Merged valid with unlabeled train'
        if include_valid:
            self._current_dict[names[0]]+= self._default_dict[names[3]]
            print 'Merged test with unlabeled train'





    def random_folds(self,k=1,valid_percent=.1):
        _n = self._default_dict['n_samples_labeled']
        _split_size = self._default_dict['split_size']
        _n_splits = len(self._default_dict[names[1]] + self._default_dict[names[2]])
        self._n_valid_samples = floor(_n*valid_percent)
        _n_valid_splits = round(self._n_valid_samples/float(_split_size))
        
        if not _valid_n_splits:
            _n_valid_split=1

        
        assert _n_splits < k * _n_valid_splits
        random.seed(123)


        _swap_idx = range(_n_splits)
        random.shuffle(_swap_idx)

        self._k_swap_idx = [ _swap_idx[i*_n_valid_splits:(i+1)*_n_valid_splits] for i in range(k) ]
        
        _folds = []
        for i in range(k):
            _c_fold = copy(self._default_dict)
            _c_fold[names[1]] += _c_fold[names[2]]
            _swap_elements = [ _c_fold[names[1]][i] for i in _swap_idx[k] ]
            _c_fold[names[2]] = _swap_elements
            for e in _swap_elements:
                _c_fold[names[1]].remove(e)

            _folds += [ _c_fold ]
        
        self._fold_dicts = _folds
        
            


    def set_fold(self,i):
        self._current_fold = i
        self._current_dict = self._fold_dicts[i]



    def set_predef_folds(dict_list_path):
        self._fold_dicts = load(open(dict_list_path))


    def get_sample_size(self):
        try:
            return self.get_train_labeled(shared_array=False)[0].shape[1]
        except:
            return self.get_train_unlabeled(shared_array=False)[0].shape[1]


    def get_label_size(self):
        try:
            _buffer = self.get_train_labeled(shared_array=False)[1].shape
            if len(_buffer) == 1:
                return 1
            else:
                return _buffer[1]
        except:
            _buffer = self.get_test(shared_array=False)[1].shape
            if len(_buffer) == 1:
                return 1
            else:
                return _buffer[1]


    def get_train_unlabeled(self,index=0,shared_array=True,cast=(None,None)):
        _n = len(self._default_dict[names[0]])

        if not _n:
            print 'no unlabeled data found.'
            return self.get_train_labeled(index,shared_array,cast)

        if (index / _n) > 0:
            warn("index out of range, taking the modulo as index", SyntaxWarning)
            if self._debug:
                print "(",index,"->",index%_n,")"
        self._current_unlabeled_train_idx = index % _n


        _buffer_unlabeled_train = self.__open_file(self._current_dict[names[0]][self._current_unlabeled_train_idx])

        _new_cast = []
        for e,c in zip(_buffer_unlabeled_train,cast):
            if c == None:
                _new_cast.append(e.dtype)
            else:
                _new_cast.append(c)
        cast = tuple( _new_cast )

        _buffer_unlabeled_train = tuple( [ array(e,dtype=t) for e,t in zip(_buffer_unlabeled_train,cast) ] )

        if shared_array:
            try:
                self._unlabeled_train_data[0].set_value(_buffer_unlabeled_train[0])
                self._unlabeled_train_data[1].set_value(_buffer_unlabeled_train[1])
                self._unlabeled_train_data[0].name = self._current_dict[names[0]][self._current_unlabeled_train_idx].split('/')[-1]+' X'
                self._unlabeled_train_data[1].name = self._current_dict[names[0]][self._current_unlabeled_train_idx].split('/')[-1]+' Y'
            except:
                self._unlabeled_train_data = (shared(value=_buffer_unlabeled_train[0],name=self._current_dict[names[0]][self._current_unlabeled_train_idx].split('/')[-1]+' X'),
                                            shared(value=_buffer_unlabeled_train[1],name=self._current_dict[names[0]][self._current_unlabeled_train_idx].split('/')[-1]+' Y'))
        else:
            self._unlabeled_train_data = _buffer_unlabeled_train

        return self._unlabeled_train_data



    def get_train_labeled(self,index=0,shared_array=True,cast=(None,None)):
        _n = len(self._default_dict[names[1]])
        if not _n:
            print 'no train data found.'
            return

        if (index / _n) > 0:
            warn("index out of range, taking the modulo as index", SyntaxWarning)
            if self._debug:
                print "(",index,"->",index%_n,")"
        self._current_labeled_train_idx = index % _n


        _buffer_train = self.__open_file(self._current_dict[names[1]][self._current_labeled_train_idx])

        _new_cast = []
        for e,c in zip(_buffer_train,cast):
            if c == None:
                _new_cast.append(e.dtype)
            else:
                _new_cast.append(c)
                cast = tuple( _new_cast )


        _buffer_train = tuple( [ array(e,dtype=t) for e,t in zip(_buffer_train,cast) ] )

        if shared_array:
            try:
                self._labeled_train_data[0].set_value(_buffer_train[0])
                self._labeled_train_data[1].set_value(_buffer_train[1])
                self._labeled_train_data[0].name = self._current_dict[names[1]][self._current_labeled_train_idx].split('/')[-1]+' X'
                self._labeled_train_data[1].name = self._current_dict[names[1]][self._current_labeled_train_idx].split('/')[-1]+' Y'
            except:
                self._labeled_train_data = (shared(value=_buffer_train[0],name=self._current_dict[names[1]][self._current_labeled_train_idx].split('/')[-1]+' X'),
                                            shared(value=_buffer_train[1],name=self._current_dict[names[1]][self._current_labeled_train_idx].split('/')[-1]+' Y'))
        else:
            self._labeled_train_data = _buffer_train

        return self._labeled_train_data

    def get_valid(self,index=0,shared_array=True,cast=(None,None)):
        _n = len(self._default_dict[names[2]])

        if not _n:
            print 'no valid data found.'
            warn("returning 10 samples from test data instead to prevent crash", SyntaxWarning)
            _test = self.get_test(index=index,shared_array=shared_array,cast=cast)
            return (_test[0][:10],_test[1][:10])
        
        if (index / _n) > 0:
            warn("index outx of range, taking the modulo as index", SyntaxWarning)
        if self._debug:
               print "(",index,"->",index%_n,")"
        self._current_valid_idx = index % _n


        _buffer_valid = self.__open_file(self._current_dict[names[2]][self._current_valid_idx])

        _new_cast = []
        for e,c in zip(_buffer_valid,cast):
            if c == None:
                _new_cast.append(e.dtype)
            else:
                _new_cast.append(c)
                cast = tuple( _new_cast )

        _buffer_valid = tuple( [ array(e,dtype=t) for e,t in zip(_buffer_valid,cast) ] )

        if shared_array:
            try:
                self._valid_data[0].set_value(_buffer_valid[0])
                self._valid_data[1].set_value(_buffer_valid[1])
                self._valid_data[0].name = self._current_dict[names[2]][self._current_valid_idx].split('/')[-1]+' X'
                self._valid_data[1].name = self._current_dict[names[2]][self._current_valid_idx].split('/')[-1]+' Y'
            except:
                self._valid_data = (shared(value=_buffer_valid[0],name=self._current_dict[names[2]][self._current_valid_idx].split('/')[-1]+' X'),
                                            shared(value=_buffer_valid[1],name=self._current_dict[names[2]][self._current_valid_idx].split('/')[-1]+' Y'))
        else:
            self._valid_data = _buffer_valid

        return self._valid_data


    def get_test(self,index=0,shared_array=True,cast=(None,None)):
        _n = len(self._default_dict[names[3]])

        if not _n:
            print 'no test data found.'
            return

        if (index / _n) > 0:
            warn("index out of range, taking the modulo as index", SyntaxWarning)
            if self._debug:
                print "(",index,"->",index%_n,")"
        self._current_test_idx = index % _n


        _buffer_test = self.__open_file(self._current_dict[names[3]][self._current_test_idx])

        _new_cast = []
        for e,c in zip(_buffer_test,cast):
            if c == None:
                _new_cast.append(e.dtype)
            else:
                _new_cast.append(c)
                cast = tuple( _new_cast )


        _buffer_test = tuple( [ array(e,dtype=t) for e,t in zip(_buffer_test,cast) ] )

        if shared_array:
            try:
                self._test_data[0].set_value(_buffer_test[0])
                self._test_data[1].set_value(_buffer_test[1])
                self._test_data[0].name = self._current_dict[names[3]][self._current_test_idx].split('/')[-1]+' X'
                self._test_data[1].name = self._current_dict[names[3]][self._current_test_idx].split('/')[-1]+' Y'
            except:
                self._test_data = (shared(value=_buffer_test[0],name=self._current_dict[names[3]][self._current_test_idx].split('/')[-1]+' X'),
                                   shared(value=_buffer_test[1],name=self._current_dict[names[3]][self._current_test_idx].split('/')[-1]+' Y'))
        else:
            self._test_data = _buffer_test

        return self._test_data


    def get_all_data_length(self):
        return sum([len(self._default_dict[name]) for name in names])
    def get_train_unlabeled_length(self):
        return len(self._current_dict[names[0]])
    def get_train_labeled_length(self):
        return len(self._current_dict[names[1]])
    def get_valid_length(self):
        return len(self._current_dict[names[2]])
    def get_test_length(self):
        return len(self._current_dict[names[3]])


    def get_all_train_unlabeled(self,shared_array=True,cast=(config.floatX,config.floatX)):
        _n = self.get_train_unlabeled_length()
        if _n == 1:
            return self.get_train_unlabeled(shared_array=shared_array,cast=cast)

        _all_train_unlabeled_x = []
        _all_train_unlabeled_y = []
        for i in range(_n):
            _train_unlabeled_x,_train_unlabeled_y = self.get_train_unlabeled(index=i,shared_array=False,cast=cast)
            _all_train_unlabeled_x.append(_train_unlabeled_x)
            _all_train_unlabeled_y.append(_train_unlabeled_y)

        _all_train_unlabeled_x = concatenate(_all_train_unlabeled_x,axis=0)
        _all_train_unlabeled_y = concatenate(_all_train_unlabeled_y,axis=0)

        if shared_array:
            return (shared(value=_all_train_unlabeled_x,name='all_train_unlabeled X'),
                    shared(value=_all_train_unlabeled_y,name='all_train_unlabeled Y'))
        else:
            return (_all_train_unlabeled_x,_all_train_unlabeled_y)


    def get_all_train_labeled(self,shared_array=True,cast=(config.floatX,config.floatX)):
        _n = self.get_train_labeled_length()
        if _n == 1:
            return self.get_train_labeled(shared_array=shared_array,cast=cast)

        _all_train_labeled_x = []
        _all_train_labeled_y = []
        for i in range(_n):
            _train_labeled_x,_train_labeled_y = self.get_train_labeled(index=i,shared_array=False,cast=cast)
            _all_train_labeled_x.append(_train_labeled_x)
            _all_train_labeled_y.append(_train_labeled_y)

        _all_train_labeled_x = concatenate(_all_train_labeled_x,axis=0)
        _all_train_labeled_y = concatenate(_all_train_labeled_y,axis=0)

        if shared_array:
            return (shared(value=_all_train_labeled_x,name='all_train_labeled X'),
                    shared(value=_all_train_labeled_y,name='all_train_labeled Y'))
        else:
            return (_all_train_labeled_x,_all_train_labeled_y)



    def get_all_valid(self,shared_array=True,cast=(config.floatX,config.floatX)):
        _n = self.get_valid_length()
        if _n == 1:
            return self.get_valid(shared_array=shared_array,cast=cast)
        elif _n == 0:
            warn("No valid defined for this dataset, using testset instead", SyntaxWarning)
            return self.get_test(shared_array=shared_array,cast=cast)

        _all_valid_x = []
        _all_valid_y = []
        for i in range(_n):
            _valid_x,_valid_y = self.get_valid(index=i,shared_array=False,cast=cast)
            _all_valid_x.append(_valid_x)
            _all_valid_y.append(_valid_y)

        _all_valid_x = concatenate(_all_valid_x,axis=0)
        _all_valid_y = concatenate(_all_valid_y,axis=0)
            

        if shared_array:
            return (shared(value=_all_valid_x,name='all_valid X'),
                    shared(value=_all_valid_y,name='all_valid Y'))
        else:
            return (_all_valid_x,_all_valid_y)


    def get_all_test(self,shared_array=True,cast=(config.floatX,config.floatX)):
        _n = self.get_test_length()
        if _n == 1:
            return self.get_test(shared_array=shared_array,cast=cast)

        _all_test_x = []
        _all_test_y = []
        for i in range(_n):
            _test_x,_test_y = self.get_test(index=i,shared_array=False,cast=cast)
            _all_test_x.append(_test_x)
            _all_test_y.append(_test_y)

        _all_test_x = concatenate(_all_test_x,axis=0)
        _all_test_y = concatenate(_all_test_y,axis=0)

        if shared_array:
            return (shared(value=_all_test_x,name='all_test X'),
                    shared(value=_all_test_y,name='all_test Y'))
        else:
            return (_all_test_x,_all_test_y)



    



    def get_all_data(self,index=0,shared_array=True,cast=(config.floatX,config.floatX)):
        _all_n = [len(self._default_dict[name]) for name in names]
        _n = sum(_all_n)

        if not _n:
            print 'no test data found.'
            return

        if (index / _n) > 0:
            warn("index out of range, taking the modulo as index", SyntaxWarning)
            print "(",index,"->",index%_n,")"
        self._current_all_idx = index % _n

        _cum_n = [ sum(_all_n[:i+1]) for i in range(len(_all_n)) ]
        
        if self._current_all_idx < _cum_n[0]:
            return self.get_train_unlabeled(index=self._current_all_idx,shared_array=shared_array,cast=cast)
        elif _cum_n[0] <= self._current_all_idx < _cum_n[1]:
            return self.get_train_labeled(index=(self._current_all_idx - _cum_n[0]),shared_array=shared_array,cast=cast)
        elif _cum_n[1] <= self._current_all_idx < _cum_n[2]:
            return self.get_valid(index=(self._current_all_idx - _cum_n[1]),shared_array=shared_array,cast=cast)
        else:
            return self.get_test(index=(self._current_all_idx - _cum_n[2]),shared_array=shared_array,cast=cast)


    def get_all_labeled_train(self):
        pass



    def get_all_unlabeled_train(self):
        pass


    def __add__(self, dataset):
        '''
        merges another dataset to the current one:
        dataset = dataset.cifar10_unsup() + dataset.cifar100_unsup() + dataset.stl_unsup()
        '''
        pass
    def get_prepro_dict(self):
        pass


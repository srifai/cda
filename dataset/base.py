import cPickle as cpkl
import gc
from os import mkdir,path,remove
from numpy import load,save,array,vstack,float32,float64,random,nan,prod,ceil,floor,isnan,hstack,ndarray,copy,asarray
from csv import reader
import pdb
def create_files(train_filelist=[],
                 valid_filelist=[],
                 test_filelist=[],
                 output_path='.',
                 max_memory=200,
                 samples_per_file=0,
                 prepro_fct=None,
                 output_type='pkl',
                 output_prefix='',
                 down_cast_float64=True,
                 shuffle_train=True,
                 shuffle_valid=True,
                 shuffle_test=True,
                 seed=0
                 ):
    '''
    input: lists of file for train, valid and test 
    output: store arrays of size max_memory in output_type format
    '''


    random.seed(seed)


    if not path.exists(output_path):
        print '--- Directory non-existent, creating it'
        mkdir(output_path)


    def open_all_files(file_path):
        print file_path
        if type(file_path) == ndarray:
            return file_path
        try:
            extension = file_path.split('.')[-1]
        except:
            import pdb; pdb.set_trace()
        if extension == 'npy':
            return load(file_path)
        elif extension == 'pkl':
            return cpkl.load(open(file_path))
        elif extension == 'csv':
            _reader = reader(open(file_path))                    
            _samples = vstack([ array(line,dtype=float32) for line in _reader ])
            return _samples
        else:
            raise ValueError('The file type has not been recognized.')


    def load_file(filetuple,down_cast_float64=True):
        _arr_x = open_all_files(filetuple[0])
        if len(filetuple)>1:
            _arr_y = open_all_files(filetuple[1])
            if _arr_y.ndim == 1:
                _arr_y = _arr_y.reshape((-1,1))

        else:
            _arr_y = array([nan]*len(_arr_x),dtype=float32).reshape((-1,1))
        type = _arr_x.dtype
        assert len(_arr_x) == len(_arr_y)
        if type == float64 and down_cast_float64:
            return (array(_arr_x,dtype=float32),_arr_y)
        else:
            return (_arr_x,_arr_y)


      



    print '--- Loading train files'
    _train_label_arrays = []
    _train_arrays = []
    for  file in train_filelist:
        if type(file) == str:
            print '------ Loading '+file[0]+'...',
        try:
            _train_struct = load_file(file,down_cast_float64)
            if isnan(_train_struct[1].sum()):
                _train_arrays.append(_train_struct)
            else:
                _train_label_arrays.append(_train_struct)
            print 'Ok!'
        except ValueError,IOError:
            print 'Failed, Skipped!'
        

    _valid_arrays = []
    print '--- Loading valid files'
    for  file in valid_filelist:
        if type(file) == str:
            print '------ Loading '+file[0]+'...',
        try:
            _valid_arrays.append(load_file(file,down_cast_float64))
            print 'Ok!'
        except ValueError,IOError:
            print 'Failed, Skipped!'


    _test_arrays = []
    print '--- Loading test files'
    for  file in test_filelist:  
        if type(file) == str:
            print '------ Loading '+file[0]+'...',
        try:
            _test_arrays.append(load_file(file,down_cast_float64))
            print 'Ok!'
        except ValueError,IOError:
            print 'Failed, Skipped!'


    _all_arrays = [ _train_arrays , _train_label_arrays , _valid_arrays , _test_arrays ]
    _names = ['train_unlabeled','train_labeled','valid','test']

    if max_memory == -1:
        _file_dict = {}
        for _name in _names:
            _file_dict[_name] = []
        for _name,arr in zip(_names,_all_arrays):
            for i,a in enumerate(arr):
                _filesdict = output_prefix+'_'+_name+str(i)
                _filename = path.join(output_path,output_prefix+'_'+_name+str(i))
                cpkl.dump(a,open(_filename+'.pkl','w'),protocol=-1)
                _file_dict[_name].append(_filesdict+'.pkl')
        cpkl.dump(_file_dict,open(path.join(output_path,output_prefix+'_files_dict.pkl'),'w'))        
        return


    
    _final_arrays = []
    for _name,arr in zip(_names,_all_arrays):
        if len(arr):
            _x = vstack([ _arr[0] for _arr in arr ])
            _y = vstack([ _arr[1] for _arr in arr ])
            
            #shuffle the arrays independantly
            if (_name == 'train_labeled' or _name == 'train_unlabeled') and shuffle_train==True:
                print '--- Shuffling '+_name+' data'
                _idx = asarray(range(len(_x)) )
                random.shuffle(_idx)
                
                #np bug
                _x2=copy(_x)
                _y2=copy(_y)
                nb=_idx.shape[0]/1000000
                
                for k in range(nb+1):
                    min_i=k*1000000
                    max_i=min((k+1)*1000000,_idx.shape[0])
                    _x2[min_i:max_i]=_x[_idx[min_i:max_i]]
                    _y2[min_i:max_i]=_y[_idx[min_i:max_i]]
                
                
                _x=_x2
                _y=_y2
                
                # old buggy code
                #_x,_y = (_x[_idx],_y[_idx])
                

            elif _name.find('valid')>-1 and shuffle_valid==True:
                print '--- Shuffling '+_name+' data'
                _idx = range(len(_x)) 
                random.shuffle(_idx)
                _x,_y = (_x[_idx],_y[_idx])


            elif _name.find('test')>-1 and shuffle_test==True:
                print '--- Shuffling '+_name+' data'
                _idx = range(len(_x)) 
                random.shuffle(_idx)
                _x,_y = (_x[_idx],_y[_idx])

            _final_arrays.append((_x,_y))

        else:
            _final_arrays.append(())
    print '--- Freeing some memory...'
    del _all_arrays
    gc.collect()
    
    _file_dict = {}
    _prepro_dict = {}
    print '--- Preprocessing...',
    if prepro_fct:
        print '--- Applying preprocessing to data and labels',
        _final_arrays_crude = prepro_fct(_names,_final_arrays)
        if type(_final_arrays_crude) == type(()):
            _final_arrays,_preprotuple = _final_arrays_crude
            _prepro_dict[_preprotuple[0]] = _preprotuple[1]
        else:
            _final_arrays = _final_arrays_crude


        _file_dict['prepro'] = prepro_fct

    print '\n--- Now dividing arrays and saving them to'+output_path+'...'
    for _name in _names:
        _file_dict[_name] = []
    for _name,_arr_tuple in zip(_names,_final_arrays):
        if len(_arr_tuple) == 0:
            continue
        print '------ ' +_name+'...',
        _x = _arr_tuple[0]
        _y = _arr_tuple[1]
        if not samples_per_file:
            _k = int(floor(_x.itemsize*prod(_x.shape)/(max_memory*10**6)))
            if _k == 0:
                _chunk = len(_x)
            else:
                _chunk = int(len(_x)/_k)
        else:
            _chunk = samples_per_file
            _k = int(len(_x)/_chunk)

        print 'making '+ str(_k+1) +' files...',
        for i in range(_k+1):
            _chunk_tuple = (_x[_chunk*i:_chunk*(i+1)],_y[_chunk*i:_chunk*(i+1)])
            if len(_chunk_tuple[0]):
                _filesdict = output_prefix+'_'+_name+str(i)
                _filename = path.join(output_path,output_prefix+'_'+_name+str(i))
                if output_type == 'pkl':
                    cpkl.dump(_chunk_tuple,open(_filename+'.pkl','w'),protocol=-1)
                    _file_dict[_name].append(_filesdict+'.pkl')
                elif output_type == 'npy':
                    _file_dict[_name].append(_filesdict+'.npy')
                    save(_filename+'.npy',_chunk_tuple[0])
                else:
                    raise('--- Output type %s not recognized'%output_type)

        print 'Ok!'

    cpkl.dump(_file_dict,open(path.join(output_path,output_prefix+'_files_dict.pkl'),'w'))
    if len(_prepro_dict.keys()):
        cpkl.dump(_file_dict,open(path.join(output_path,output_prefix+'_prepro_dict.pkl'),'w'))

    print 'All finished.'
    return path.join(output_path,output_prefix+'_files_dict.pkl')


def transform_dataset(dataset,
                      
 
                      train_prepro_x=None,
                      train_prepro_y=None,
                      valid_prepro_x=None,
                      valid_prepro_y=None,
                      test_prepro_x=None,
                      test_prepro_y=None,

                      train_data_label_transform=None,
                      valid_data_label_transform=None,
                      test_data_label_transform=None,

                      
                      output_path='.',
                      max_memory=200,
                      samples_per_file=0,
                      prepro_fct=None,
                      output_type='pkl',
                      output_prefix='',
                      down_cast_float64=True,
                      shuffle=False):

    '''
    This function can be used to transform a dataset statically instead of dynamically when
    the preoprocessing function takes to much memory or time to execute on the fly

    Two types of prepro function can be used:

    1) the same prepro function as in the dynamic preprocessing. In this case, set the following parameters:
          train_prepro_x=None,
          train_prepro_y=None,
          valid_prepro_x=None,
          valid_prepro_y=None,
          test_prepro_x=None,
          test_prepro_y=None,

    2) global preprocessing functions that transform both the labels and the data at the same time. (for example, if you want to use a subset of the dataset)
       In that case set the following parameters:
          train_data_label_transform_fcn=None,
          valid_data_label_transform_fcn=None,
          test_data_label_transform_fcn=None,

    
    
    The function will call the create_files function to create a new dataset and therefore recieves as parameters all the settings necessary for a call to create_files

    

    Limitations: This function cannot apply a preprocessing that is dependant on the whole dataset simultaneously. This should be coded in a another function
    as the condisederations are readicially different (for example if you want to calculate the principal compenents of the dataset).
    '''

    train_filelist=[]
    valid_filelist=[]
    test_filelist=[]

    
    id=random.randint(0,15000)
    id=1500 #debug
    # treat train first

    
    for i in xrange(dataset.get_train_labeled_length()):
        print 'preprocessing train file ' + str(i)
        _train_x,_train_y=dataset.get_train_labeled(index=i,shared_array=False)

        if train_prepro_x!=None:
            _ptrain_x=train_prepro_x(_train_x)
        else:
            _ptrain_x=_train_x


        if train_prepro_y!=None:
            _ptrain_y=train_prepro_y(_train_y)
        else:
            _ptrain_y=_train_y

        if train_data_label_transform!=None:
            _pptrain_x,_pptrain_y=train_data_label_transform(_ptrain_x,_ptrain_y)
        else:
            _pptrain_x=_ptrain_x
            _pptrain_y=_ptrain_y
        
        # create a temporary file 
        fname='temp_train_data'+str(id)+'_'+str(i)+'.npy'
        trainx_name= path.join(output_path,fname)
        fname='temp_train_label'+str(id)+'_'+str(i)+'.npy'
        trainy_name= path.join(output_path,fname)
        train_filelist+=[(trainx_name,trainy_name)]
        save(trainx_name,_pptrain_x)
        save(trainy_name,_pptrain_y)
        

    for i in xrange(dataset.get_valid_length()):
        print 'preprocessing valid file ' + str(i)
        _valid_x,_valid_y=dataset.get_valid(index=i,shared_array=False)

        if valid_prepro_x!=None:
            _pvalid_x=valid_prepro_x(_valid_x)
        else:
            _pvalid_x=_valid_x


        if valid_prepro_y!=None:
            _pvalid_y=valid_prepro_y(_valid_y)
        else:
            _pvalid_y=_valid_y

        if valid_data_label_transform!=None:
            _ppvalid_x,_ppvalid_y=valid_data_label_transform(_pvalid_x,_pvalid_y)
        else:
            _ppvalid_x=_pvalid_x
            _ppvalid_y=_pvalid_y

        # create a temporary file 
        fname='temp_valid_data'+str(id)+'_'+str(i)+'.npy'
        validx_name= path.join(output_path,fname)
        fname='temp_valid_label'+str(id)+'_'+str(i)+'.npy'
        validy_name= path.join(output_path,fname)
        valid_filelist+=[(validx_name,validy_name)]
        save(validx_name,_ppvalid_x)
        save(validy_name,_ppvalid_y)


    for i in xrange(dataset.get_test_length()):
        print 'preprocessing test file ' + str(i)
        _test_x,_test_y=dataset.get_test(index=i,shared_array=False)

        if test_prepro_x!=None:
            _ptest_x=test_prepro_x(_test_x)
        else:
            _ptest_x=_test_x


        if test_prepro_y!=None:
            _ptest_y=test_prepro_y(_test_y)
        else:
            _ptest_y=_test_y

        if test_data_label_transform!=None:
            _pptest_x,_pptest_y=test_data_label_transform(_ptest_x,_ptest_y)
        else:
            _pptest_x=_ptest_x
            _pptest_y=_ptest_y

        # create a temporary file 
        fname='temp_test_data'+str(id)+'_'+str(i)+'.npy'
        testx_name= path.join(output_path,fname)
        fname='temp_test_label'+str(id)+'_'+str(i)+'.npy'
        testy_name= path.join(output_path,fname)
        test_filelist+=[(testx_name,testy_name)]
        save(testx_name,_pptest_x)
        save(testy_name,_pptest_y)

    
    # create the new dataset:
    create_files(train_filelist=train_filelist,
                 valid_filelist=valid_filelist,
                 test_filelist=test_filelist,
                 output_path=output_path,
                 max_memory=max_memory,
                 samples_per_file=samples_per_file,
                 prepro_fct=prepro_fct,
                 output_type=output_type,
                 output_prefix=output_prefix,
                 down_cast_float64=down_cast_float64,
                 shuffle=shuffle)


    # erase the temporary files
    
    for tuple in train_filelist:
        remove(tuple[0])
        remove(tuple[1])
    
    for tuple in valid_filelist:
        remove(tuple[0])
        remove(tuple[1])

    for tuple in test_filelist:
        remove(tuple[0])
        remove(tuple[1])


                 
    
        
    
        
        
        

    
                      
                      

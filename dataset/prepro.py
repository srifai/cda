''' The preprocessing function should follow
the next template'''
import pdb
from numpy import float32
#from sklearn.feature_extraction import image
#from sklearn.decomposition.pca import PCA

def template(_names, _final_arrays):
    '''in general _names is [train_unsupervised, train,valid,test]
    and _final_arrays contains the corresponding _x and _y.
    the function has to output the _final_arrays
    '''
    def preproc_fct_x(x):
        ''' write your preproc function here'''
        return x

    def preproc_fct_y(y):
        ''' write your preproc function here'''
        return y

    _new_final_arrays = []
    for _name,_arr_tuple in zip(_names,_final_arrays):
        print '\n------ ' +_name+'...',
        if len(_arr_tuple)==0:
            print 'No data in here!',
            _new_final_arrays.append(_arr_tuple)
        else:
            print '--- Applying preprocessing to data',
            _x = _arr_tuple[0]
            _arr_tuple[0] = preproc_fct_x(_x)
            print '--- Applying preprocessing to labels',
            _y = _arr_tuple[1]
            _arr_tuple[1] = preproc_fct_y(_y)
            _new_final_arrays.append(_arr_tuple)
    
    return _new_final_arrays

def minmax(_names, _final_arrays):
    '''rescale the input between 0 and 1 by subtracting the min
    and dividing by the max, cast the returned array to float32'''
    
    def preproc_fct_x(x):
        x = float32(x)
        x -= x.min()
        x /= x.max()
        return x

    def preproc_fct_y(y):
        return y

    _new_final_arrays = []
    for _name,_arr_tuple in zip(_names,_final_arrays):
        print '\n------ ' +_name+'...',
        if len(_arr_tuple)==0:
            print 'No data in here!',
            _new_final_arrays.append(_arr_tuple)
        else:
            print '--- Applying preprocessing to data',
            _x = _arr_tuple[0]
            _prepro_x = preproc_fct_x(_x)
            print '--- Applying preprocessing to labels',
            _y = _arr_tuple[1]
            _prepro_y = preproc_fct_y(_y)
            _new_final_arrays.append((_prepro_x, _prepro_y))
    
    return _new_final_arrays


def coates_pipeline(_names, _final_arrays, 
                    lcn=True, 
                    pca=True,
                    whiten=True,
                    components=80,
                    drop_component=1,
                    patch_size=(8,8),
                    center_mode='all'):


    def lcn(patches):
        if center_mode == 'all':
            # center all colour channels together
            patches = patches.reshape((patches.shape[0], -1))
            patches -= patches.mean(axis=1)[:,None]
        elif center_mode == 'channel':
            #center each colour channel individually
            patches -= patches.mean(axis=2).mean(axis=1).reshape((n_patches, 1, 1, 3))
            patches = patches.reshape((n_patches, -1))
        elif center_mode == 'none':
            # do not center the pixels
            patches = patches.reshape((n_patches, -1))
        else:
            assert False

        patches_std = np.sqrt((patches**2).mean(axis=1))
        min_divisor = (2*patches_std.min() + patches_std.mean()) / 3
        patches /= np.maximum(min_divisor, patches_std).reshape((patches.shape[0],1))
        return patches


    def fit_pca(images):
        """Fit the feature extractor on a collection of 2D images"""
        print '--- Extracting patches'
        for im in images:
            pdb.set_trace()
            patch = image.extract_patches_2d(im,patch_size)
            patches.append(patch)#.reshape((1,len(patch))+patch_size))
        patches = numpy.concatenate(patches,axis=0)
        n_patches = patches.shape[0]
        if lcn:
            patches = lcn(patches)

        pca = PCA(whiten=whiten, n_components=components)
        pca.fit(patches)
        patches_pca = pca.transform(patches)
        if n_drop_components:
            pca.components_[:,:drop_components] = 0
            pdb.set_trace()
            patches_pca[:,:drop_components] = 0

        return (patches_pca,pca)

    def preproc_fct_x(x):
        return fit_pca(x)

    def preproc_fct_y(y):
        ''' write your preproc function here'''
        return y


    _new_final_arrays = []
    _x,pca = preproc_fct_x(_final_arrays[0][0])
    _new_final_arrays = [ (_x,_final_arrays[0][1]) ] + [ () for i in range(len(_final_arrays)-1) ]

    return _new_final_arrays,('pca',pca)

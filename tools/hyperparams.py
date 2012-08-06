import numpy as np
import pdb




class hyperparams(object):
    '''
    the class recieves a list of triplets [(type=string,hp_name=string,values=list )]
    
    type: 3 choices 1) 'values': values will contain the exact hp to use. those values will be used in the cartesian product of all hp values
                    2) 'uniform': values will contain a tuple (min_val,max_val)]. a uniform random will be used to generate the values between min and max
                                  the values chosen will either be ints or floats depeneding on the min_val and max_val types
                    3) 'random elem' : values will contain a list  [possible values] and the values will be chosen randomly from possible values


    example usage:

    hp=hyperparams( [('values','net_size',[1000,2000]),
                     ('uniform','lr',[0.05,0.25]),
                     ('random element','bs',[5,10,20,40]),
                    ]
                  )
    jobs=hp.create_hp(20)
    jobs2=hp.create_hp(2)


                   
                    
    '''

    def __init__(self,hp_list):
        '''
        create the values dict and random dict
        '''
        self._values_dict={}
        self._random_dict={}
        self.hp_list=[]

        for elem in hp_list:
            self.add_hp(elem)

            



    def add_hp(self,hp_triplet):
        
        # calculate the values
        if hp_triplet[0]=='values':
            val=hp_triplet[2]
            if hp_triplet[1] not in self._values_dict:
                self._values_dict[hp_triplet[1]]=val
            else:
                temp = list(set(self._values_dict[hp_triplet[1]] + val))
                self._values_dict[hp_triplet[1]]=temp


        else:
            self._random_dict[hp_triplet[1]] =(hp_triplet[0],hp_triplet[2])
       


    def create_hp(self,nb_ex):
        
        # first, do the cartesian product of values_dict
        job_list=[{}]
        all_keys = self._values_dict.keys()
        for key in all_keys:
            possible_values = self._values_dict[key]
            new_job_list = []
            for val in possible_values:
                for job in job_list:
                    to_insert = job.copy()
                    to_insert.update({key: val})
                    new_job_list.append(to_insert)
            job_list = new_job_list

        
        #check how many experiments we have and how many are needed
        #fill the list deterministically by looping over the
        #orginal jobs
        
        nb=len(job_list)
        tnb=nb
        while tnb<nb_ex:
            #idx=np.random.randint(0,nb)
            temp=job_list[tnb%nb].copy()
            job_list.append(temp)
            tnb+=1

        
        # generate all the random values and add them to each job
        for k,v in self._random_dict.iteritems():
            if v[0]=='uniform':
                values=self._generate_uniform_values(v[1][0],v[1][1],max(nb_ex,nb))
            elif v[0]=='random element':
                values=self._generate_random_samples(v[1],max(nb_ex,nb))
            else:
                print " ERROR " + v[0] + ' is not a supported type'

            for i in xrange(len(job_list)):
                job_list[i][k]=values[i]


        
        self.job_list=job_list
        return job_list
        
     

    def _generate_uniform_values(self,min,max,nb):
        if type(min)==float or type(max)==float:
            values=(max-min)*np.random.ranf(size=(nb,))+min
        else:
            values=np.random.random_integers(min,high=max,size=(nb,))
        return values.tolist()

    def _generate_random_samples(self,elems,nb):
        
        range=len(elems)
        idx=np.random.randint(0,high=range,size=(nb,))
        return np.asarray(elems)[idx].tolist()
        



#temp
def hl(start_index=(0,0),length=32,shape=(32,32)):

    temp=np.zeros(shape,dtype='float32')
    temp[start_index[0],start_index[1]:start_index[1]+length]=1
    return temp.flatten()


def vl(start_index=(0,0),length=32,shape=(32,32)):
    temp=np.zeros(shape,dtype='float32')
    temp[start_index[0]:start_index[0]+length,start_index[1]]=1
    return temp.flatten()


def box(start_index=(0,0),size=(16,16),shape=(32,32)): #size is heigth width
    temp=np.zeros(shape[0]*shape[0],dtype='float32')
    temp+=hl(start_index=start_index,length=size[1],shape=shape)
    temp+=hl(start_index=(start_index[0]+size[0]-1,start_index[1]),length=size[1],shape=shape)
    
    temp+=vl(start_index=start_index,length=size[0],shape=shape)
    temp+=vl(start_index=(start_index[0],start_index[1]+size[1]-1),length=size[0],shape=shape)
    temp=np.minimum(temp,1.0)
    return temp


def create_w(nb_filters=1024,shape=(32,32)):
    w=np.zeros((nb_filters,shape[0]*shape[1]),dtype='float32')
    current_idx=0
    for i in xrange(32):
        w[current_idx]=hl((i,0),32,shape)
        current_idx+=1
    for i in xrange(32):
        w[current_idx]=vl((0,i),32,shape)
        current_idx+=1

    
    for i in xrange(15):
        w[current_idx]=box((i,i),(32-2*i,32-2*i),shape)
        current_idx+=1
        
    for i in xrange(7):
        w[current_idx]=box((2*i,i),(32-4*i,32-2*i),shape)
        current_idx+=1

    for i in xrange(7):
        w[current_idx]=box((i,2*i),(32-2*i,32-4*i),shape)
        current_idx+=1

    for i in xrange(7):
        for j in xrange(7):
            w[current_idx]=box((7+i,7+2*j),(32-2*7,32-4*7),shape)
            current_idx+=1

    for i in xrange(7):
        for j in xrange(7):
            w[current_idx]=box((7+2*i,7+i),(32-4*7,32-2*7),shape)
            current_idx+=1


    for i in xrange(nb_filters-current_idx):
        nb_h=np.random.randint(0,5)
        nb_v=np.random.randint(0,5)
        
        for j in xrange(nb_h):
            w[current_idx]+=hl((np.random.randint(6,shape[0]-6),0),32,shape)

        for j in xrange(nb_v):
            w[current_idx]+=vl((0,np.random.randint(6,shape[1]-6)),32,shape)

        w[current_idx]=np.minimum(w[current_idx],1.0)


        current_idx+=1
    
                           

    return w

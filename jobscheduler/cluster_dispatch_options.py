local = {   'cluster':'--local',
            'env':'--env=THEANO_FLAGS=floatX=float32',
            }

condor = {  'cluster':'--condor',
            'env':'--env=THEANO_FLAGS=floatX=float32',
            'os' :'--os=fc14',
            'num_cores':'--repeat_jobs=10'
            }

angel = {  'cluster':'--sharcnet',
           'q':'--queue=gpu',
           'cpu':'--cpu=1',
           'gpu':'--gpu',
           'env':'--env=THEANO_FLAGS=floatX=float32,device=gpu',
           'num_cores':'--repeat_jobs=12',
           'mem':'--mem=4G'
           }

monk = {  'cluster':'--sharcnet',
           #'q':'--queue=cpu',
           'cpu':'--cpu=1',
           'gpu':'--gpu',
           'env':'--env=THEANO_FLAGS=floatX=float32,device=gpu',
           'num_cores':'--repeat_jobs=12',
           'mem':'--mem=4G'
           }


briaree = {
            'cluster':'--torque',
            'env':'--env=THEANO_FLAGS=floatX=float32',
            'num_cores':'--repeat_jobs=12'
            }

clusters = { 'condor':condor,
             'angel':angel,
             'briaree':briaree,
             'monk':monk}

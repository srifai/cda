from jobman.sql import  db, EXPERIMENT, insert_dict
from jobman import DD
import os
import sys
import string
import argparse
import cluster_dispatch_options as cdo
from configobj import ConfigObj

def run(program, args):
    for path in string.split(os.environ["PATH"], os.pathsep):
        file = os.path.join(path, program)
        try:
            return os.execv(file,('',)+args)
        except os.error:
            pass
    raise os.error, "cannot find executable"



class JobParser(object):
    def __init__(self,jobman_entrypoint,hp_sampler):
        self.je = jobman_entrypoint
        print 'sys.argv[0]:', sys.argv[0]
        print 'os.path.basename(sys.argv[0]):', os.path.basename(sys.argv[0])
        self.filename = os.path.basename(sys.argv[0]).split('.py')[0]
        print 'self.filename:', self.filename
        self.sampler = hp_sampler
        self.jobs = [DD(dic) for dic in self.sampler.create_hp(1)]
        config_file = os.getenv('HOME') + '/.polyrc'
        if os.path.exists(config_file):
            self.config = ConfigObj(os.getenv('HOME')+'/.polyrc')
        else:
            self.config = {}
        # Use an sqlite database by default
        self.dbpath = self.config.get('dbpath',
                'sqlite:///%s.sqlite' % self.filename)
        self.dbpath = self.dbpath.strip('/') + '?table=%s' % self.filename

        self.exppath = self.filename+'.jobman_entrypoint'
        self.globals = cdo.clusters[self.config.get('cluster', 'local')]
        self.parser = argparse.ArgumentParser(description='Jobman Swiss Knife utility',
                                              epilog="",
                                              prefix_chars='-+/')

        self._init_arguments = self.define_arguments()
        self.parse()

    def parse(self):
        self.args = self.parser.parse_args()
        options = vars(self.args)

        if options['gpu']:
            self.globals['env']='--env=THEANO_FLAGS=floatX=float32,device=gpu'
            self.globals['gpu']='--gpu'

        if options['memory']:
            self.globals['mem']='--mem='+str(options['memory'])

        for k,v in options.iteritems():
            if k == 'insertjobs' and v != None:
                self.insert(v)

            elif k == 'create_view' and v:
                run('jobman',('sqlview',self.dbpath,self.filename+'_view'))

            elif k == 'njobs' and v != None:
                self.dispatch(v)
 
            elif k == 'test' and v:
                self.test()

            elif k == 'query' and v:
                self.view()
            elif k == 'status' and v:
                self.reset(v)
            else:
                continue



    def test(self):
        print '--- Entering Test mode'
        chanmock = DD({'COMPLETE':0,'save':(lambda:None)})
        self.je(self.jobs[0], chanmock)
        print '--- Test Finished!'

    def dispatch(self,v=-1):
        if v != -1:
            self.globals['num_cores']='--repeat_jobs='+str(int(v))
        run('jobdispatch',tuple(self.globals.values())+('jobman','sql','-n0',self.dbpath,self.config.get('exppath', '.')))


    def insert(self,v=-1):
        print self.dbpath
        _db = db(self.dbpath)
        j = 0
        self.jobs = [DD(dic) for dic in self.sampler.create_hp(v)]
        for i,_job in enumerate(self.jobs):
            _job.update({EXPERIMENT: str(self.exppath)})
            insert_dict(_job,_db)
            j+=1
            print 'inserting job: '+str(_job)
            if i == v:
                break
        print 'Inserted',j,'jobs in',self.dbpath


    def reset(self,v=5):
        print self.dbpath
        run('jobman',('sqlstatus',self.dbpath,'--status='+str(v),'--set_status=0'))

    def define_arguments(self):
        self.parser.add_argument('-c','--cluster',
                                 help='chooses a cluster (condor,mamouth,...) for config',
                                 action='store_const',
                                 const=self.config.get('cluster', 'local'))

        self.parser.add_argument('-i','--insert',
                                 help=' inserts jobs in database. If no arguments is given, insert all available HPs.',
                                 dest='insertjobs',
                                 type=int)

        self.parser.add_argument('-m','--mem',
                                 help=' sets memory per job for dispatch.',
                                 dest='memory',
                                 type=int)
        
        self.parser.add_argument('-v','--create_view',
                                 help=' creates database view table.',
                                 action='store_true')
        
        self.parser.add_argument('-d','--dispatch',
                                 help=' dispatches the jobs inserted in the database.',
                                 dest='njobs',
                                 type=int)
        
        self.parser.add_argument('-t','--test',
                                 help=' test the experiment without dispatching it(debug).',
                                 action='store_true')

        self.parser.add_argument('-g','--gpu',
                                 help=' run the jobs on the gpu queue.',
                                 action='store_true')

        self.parser.add_argument('-q','--query',
                                 help=' queries jobman table and displays it in console.',
                                 action='store_true')

        self.parser.add_argument('-r','--reset',
                                 help=' resets the specified status to 0 in the db.',
                                 dest='status',
                                 type=int)


    def view(self):
        """
        View all the jobs in the database.
        """
        import commands
        import sqlalchemy
        import psycopg2

        # Update view
        url = self.dbpath
        commands.getoutput("jobman sqlview %s%s %s_view" % (url, self.filename, self.filename))

        # Display output
        def connect():
            return psycopg2.connect(user=self.config['dbuser'],password=self.config['dbpassword'],
                                    database=self.config['dbuser']+'_db', host=self.config['dbhost'])

        engine = sqlalchemy.create_engine('postgres://', creator=connect)
        conn = engine.connect()
        experiments = sqlalchemy.Table('%s_view' % self.filename,
                                       sqlalchemy.MetaData(engine), autoload=True)

        columns = [ experiments.columns[k] for k in experiments.columns.keys() if k.find('jobman') == -1 ] + [ experiments.columns['jobman_status'] ]
        results = sqlalchemy.select(columns).execute()
        results = [map(lambda x: x.name, columns)] + list(results)

        def get_max_width(table, index):
            """Get the maximum width of the given column index"""
            return max([len(format_num(row[index])) for row in table])

        def format_num(num):
            """Format a number according to given places.
            Adds commas, etc. Will truncate floats into ints!"""
            try:
                if "." in num:
                    return "%.7f" % float(num)
                else:
                    return int(num)
            except (ValueError, TypeError):
                return str(num)

        col_paddings = []

        for i in range(len(results[0])):
            col_paddings.append(get_max_width(results, i))

        for row_num, row in enumerate(results):
            for i in range(len(row)):
                col = format_num(row[i]).ljust(col_paddings[i] + 2) + "|"
                print col,
            print

            if row_num == 0:
                for i in range(len(row)):
                    print "".ljust(col_paddings[i] + 1, "-") + " +",
                print


class JobHandler(object):
    def __init__(self,state, channel):
        self.channel = channel
        self.state = state
        print 'state:',state
    def save(self):
        self.channel.save()



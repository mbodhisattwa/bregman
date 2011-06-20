# AudioDB - routines for audio database I/O, searching, and manipulation
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "New BSD License"
__email__ = 'mcasey@dartmouth.edu'


# AudioDB libraries
import glob
import error
import pylab
import features

try:
    import pyadb
except ImportError:
    pass
from scipy.signal import resample

class adb:
    """
    ::
    
        A helper class for handling audiodb databases
    """
    @staticmethod
    def read(fname):
        """
        ::

            read a binary little-endien row-major adb array from a file.
            Uses open, seek, fread
        """
        fd = None
        data = None
        try:
            fd = open(fname, 'rb')
            dim = pylab.np.fromfile(fd, dtype='<i4', count=1)
            data = pylab.np.fromfile(fd, dtype='<f8')
            data = data.reshape(-1,dim)
            return data
        except IOError:
            print "IOError: Cannot open %s for reading." %(fname)
            raise IOError
        finally:
            if fd:
                fd.close()

    @staticmethod
    def write(fname,data):
        """
        ::

            write a binary little-endien row-major adb array to a file.
        """
        fd = None
        rows,cols=data.shape
        try:
            fd = open(fname, 'wb')
            pylab.array([cols],dtype='<i4').tofile(fd)
            data = pylab.np.array(data,dtype='<f8')
            data.tofile(fd)
        except IOError:
            print "IOError: Cannot open %s for writing." %(fname)
            raise IOError
        finally:
            if fd:
                fd.close()

    @staticmethod
    def read_list(fileList):
        """
        ::

            read feature data from a list of filenames into a list of arrays
        """
        all_data = []
        #data = adb.read(fileList[0])
        #shape = data.shape
        for i,v in enumerate(fileList):
            all_data.append(adb.read(v))
        return all_data

    @staticmethod
    def read_dir(dir_expr):    
        """
        ::

            read directory of binary features with wilcards, e.g. dir('Features/*/*.mfcc')
            returns a list of observation matrices, one for each file in the wildcard search
        """
        fileList = glob.glob(dir_expr)
        return adb.read_list(fileList)

    @staticmethod
    def cov_vector(data):
        """
        ::

            turn time-series into Gaussian parameter vector consisting of mean and vectorized covariance
        """
        C = pylab.np.cov(data.T,rowvar=0)
        M = pylab.np.mean(data.T,1)
        features = pylab.np.r_[M, C.reshape((pylab.np.prod(C.shape)))]
        return features

    @staticmethod
    def cov_vector_file(fname):
        """
        ::

            read time-series features from a file and convert to vectorized means and covariance matrix
        """
        data = adb.read(fname)
        features = adb.cov_vector(data)
        return features

    @staticmethod
    def cov_list(fileList):
        """
        ::

            read time-series features from a file list, convert to covariance vectors and stack as a matrix    
        """
        # sniff the first file
        data = adb.read(fileList[0])
        shape = data.shape
        X = pylab.np.zeros((len(fileList),shape[1]*(shape[1]+1)))
        for i,v in enumerate(fileList):
            X[i,:] = adb.cov_vector(v)
        return X

    @staticmethod
    def cov_dir(dir_expr):    
        """
        ::

            load features from wildcard search into vectorized covariance stacked matrix, one file per row
        """
        fileList = glob.glob(dir_expr)
        return adb.cov_list(fileList)

    @staticmethod
    def resample_vector(data, prop):
        """
        ::

            resample the columns of data by a factor of prop e.g. 0.75, 1.25,...)
        """
        new_features = resample(data, pylab.around(data.shape[0]*prop))
        return new_features

    @staticmethod
    def sparseness(data):
        """
        ::

            Sparseness measure of row vector data.
            Returns a value between 0.0 (smooth) and 1.0 (impulse)
        """
        X = data
        if pylab.np.abs(X).sum() < pylab.np.finfo(pylab.np.float32).eps:
            return pylab.np.array([0.])
        r = X.shape[0]        
        s = (pylab.np.sqrt(r) - pylab.np.abs(X).sum(0)/pylab.np.sqrt((X**2).sum(0))) / (pylab.np.sqrt(r)-1)
        return s

    @staticmethod
    def get(dbname, mode="r"):
        """
        ::

            Retrieve ADB database, or create if doesn't exist
        """
        db = pyadb.Pyadb(dbname, mode=mode)
        stat = db.status()
        if not stat['l2Normed']:
            pyadb._pyadb._pyadb_l2norm(db._db)
        if not stat['hasPower']:
            pyadb._pyadb._pyadb_power(db._db)
        # make a sane configQuery for this db
        db.configQuery={'accumulation': 'track','distance': 'eucNorm','exhaustive': False,'falsePositives': False, 'npoints': 10,'ntracks': 10,'seqStart': 100, 'seqLength': 10, 'radius':0.4, 'absThres':-4.5, 'resFmt': 'list'}
        db.delta_time = 0.1 # feature delta time in seconds
        return db

    @staticmethod
    def insert(db, X, P, Key, T=None):
        """
        ::

            Place features X and powers P into the adb database with unique identifier given by string "Key"
        """
        db.insert(featData=X, powerData=P, timesData=T, key=Key)

    @staticmethod
    def search(db, Key):
        """
        ::

            Static search method
            returns sorted list of results
        """
        if not db.configCheck():
            print "Failed configCheck in query spec."
            print db.configQuery
            return None
        res = db.query(Key)
        res_resorted = adb.sort_search_result(res.rawData)
        return res_resorted

    @staticmethod
    def tempo_search(db, Key, tempo):
        """
        ::

            Static tempo-invariant search
            Returns search results for query resampled over a range of tempos.
        """
        if not db.configCheck():
            print "Failed configCheck in query spec."
            print db.configQuery
            return None
        prop = 1./tempo # the proportion of original samples required for new tempo
        qconf = db.configQuery.copy()
        X = db.retrieve_datum(Key)
        P = db.retrieve_datum(Key, powers=True)
        X_m = pylab.mat(X.mean(0))
        X_resamp = pylab.array(adb.resample_vector(X - pylab.mat(pylab.ones(X.shape[0])).T * X_m, prop))
        X_resamp += pylab.mat(pylab.ones(X_resamp.shape[0])).T * X_m
        P_resamp = pylab.array(adb.resample_vector(P, prop))
        seqStart = int( pylab.around(qconf['seqStart'] * prop) )
        qconf['seqStart'] = seqStart
        seqLength = int( pylab.around(qconf['seqLength'] * prop) )
        qconf['seqLength'] = seqLength
        tmpconf = db.configQuery
        db.configQuery = qconf
        res = db.query_data(featData=X_resamp, powerData=P_resamp)
        res_resorted = adb.sort_search_result(res.rawData)
        db.configQuery = tmpconf
        return res_resorted

    @staticmethod
    def sort_search_result(res):
        """
        ::

            Sort search results by stripping out repeated results and placing in increasing order of distance.
        """
        if not res or res==None:
            return None
        a,b,c,d = zip(*res)
        u = adb.uniquify(a)
        i = 0
        j = 0
        k = 0
        new_res=[]
        while k < len(u)-1:
            test_str = u[k+1]
            try:
                j = a.index(test_str,i)
            except ValueError:
                break
            tmp=res[i:j]
            tmp.reverse()
            for z in tmp: new_res.append(z) 
            i = j
            k += 1
        if j<len(res)-1:
            tmp=res[j:len(res)]
            tmp.reverse()
            for z in tmp: new_res.append(z) 
        return new_res

    @staticmethod
    def uniquify(seq, idfun=None): 
        """
        ::

            Remove repeated results from result list
        """
        # order preserving
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
        return result

    @staticmethod
    def insert_audio_files(fileList, dbName, chroma=True, mfcc=False, cqft=False, progress=None):
        """
        ::

            Simple insert features into an audioDB database named by dbBame.
            Features are either chroma [default], mfcc, or cqft. 
            Feature parameters are default.
        """
        db = adb.get(dbName, "w")
        if not db:
            print "Could not open database: %s" %dbName
            return False    
        del db # commit the changes by closing the header
        db = adb.get(dbName) # re-open for writing data
        # FIXME: need to test if KEY (%i) already exists in db
        # Support for removing keys via include/exclude keys
        for a, i in enumerate(fileList):
            if progress:
                progress((a+0.5)/float(len(fileList)),i) # frac, fname
            print "Processing file: %s" %i
            F = features.Features(i)            
            if chroma: F.feature_params['feature']='chroma'
            elif mfcc: F.feature_params['feature']='mfcc'
            elif cqft: F.feature_params['feature']='cqft'
            else:
                print "One of chroma, mfcc, or cqft must be specified for features"
                raise error.BregmanError()
            F.extract()
            # raw features and power in Bels
            if progress:
                progress((a+1.0)/float(len(fileList)),i) # frac, fname
            db.insert(featData=F.CHROMA.T, powerData=adb.feature_scale(F.POWER, bels=True), key=i) 
            # db.insert(featData=F.CHROMA.T, powerData=F.feature_scale(F.POWER, bels=True), key=i)
        return db


    @staticmethod
    def insert_feature_files(featureList, powerList, keyList, dbName, delta_time=None, undo_log10=False):
        """
        ::

            Walk the list of features, powers, keys, and, optionally, times, and insert into database
        """
        
        if delta_time == None:
            delta_time = 0.0
        db = adb.get(dbName,"w")

        if not db:
            print "Could not open database: %s" %dbName
            return False    
        # FIXME: need to test if KEY (%i) already exists in db
        # Support for removing keys via include/exclude keys
        for feat,pwr,key in zip(featureList, powerList, keyList):
            print "Processing features: %s" %key
            F = adb.read(feat)
            P = adb.read(pwr)
            a,b = F.shape

            if(len(P.shape)==2):
                P = P.reshape(P.shape[0])

            if(len(P.shape)==1):
                c = P.shape[0]
            else:
                print "Error: powers have incorrect shape={0}".format(P.shape)
                return None

            if a != c:
                F=F.T
                a,b = F.shape
                if a != c:
                    print "Error: powers and features different lengths powers={0}*, features={1},{2}*".format(c, a, b)
                    return None
            # raw features, power in Bels, and key
            if undo_log10:
                F = 10**F
            if delta_time!=0.0:
                T = pylab.c_[pylab.arange(0,a)*delta_time, (pylab.arange(0,a)+1)*delta_time].reshape(1,2*a).squeeze() 
            else:
                T = None
            db.insert(featData=F, powerData=P, timesData=T, key=key)
        return db

    @staticmethod
    def normalize(x):
        """
        ::

            static method to copy array x to new array with min 0.0 and max 1.0
        """
        y=x.copy()
        y=y-pylab.np.min(y)
        y=y/pylab.np.max(y)
        return y

    @staticmethod
    def feature_plot(M, normalize=False, dbscale=False, norm=False, title_string=None, interp='nearest', bels=False):
        """
        ::

            static method for plotting a matrix as a time-frequency distribution (audio features)
        """
        X = adb.feature_scale(M, normalize, dbscale, norm, bels)
        pylab.figure()
        clip=-100.
        if dbscale or bels:
            if bels: clip/=10.
            pylab.imshow(pylab.clip(X,clip,0),origin='lower',aspect='auto', interpolation=interp)
        else:
            pylab.imshow(X,origin='lower',aspect='auto', interpolation=interp)
        if title_string:
            pylab.title(title_string)
        pylab.colorbar()

    @staticmethod
    def feature_scale(M, normalize=False, dbscale=False, norm=False, bels=False):
        """
        ::
        
            Perform mutually-orthogonal scaling operations, otherwise return identity:
              normalize [False]
              dbscale  [False]
              norm      [False]        
        """
        if not (normalize or dbscale or norm or bels):
            return M
        else:
            X = M.copy() # don't alter the original
            if norm:
                X = X / pylab.tile(pylab.sqrt((X*X).sum(0)),(X.shape[0],1))
            if normalize:
                X = adb.normalize(X)
            if dbscale or bels:
                X = pylab.log10(X)
                if dbscale:                
                    X = 20*X
        return X



import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted
import unicodedata

class FeatEncoder(TransformerMixin):
    """
    Class which detect automatically data types and rename the columns with a prefix to denotate the type, with the following mapping:
    
    bin for binary  
    cont for continue,
    ord for ord,
    txt for text,
    date for datetime and timedelta,
    other for other types of objects
    
    It is possible to specify some features in the category which you want, and it will classify the rest of the features
    
    Returns
    -------
    DataFrame
        Dataframe with columns renamed
    """
    
    def __init__(self,feats_tgt=[], feats_id=[], feats_cont=[], feats_str=[], feats_bin=[],
                  feats_ord=[], feats_date=[], feats_other=[], feats_cluster=[], bin_transform=True,
                 cardinal_threshold=5, symbols_nothing=[',','.', ';', "'", '´','*', '~'], 
                 symbols_underscore=[' ','|', '/', '\\'], method=None):
        
        """ 
        feats_tgt: list
            List of features of target feature, default []
        feats_id: list
            List of features of type id, default []
        feats_cont: list
            List of features of type continue, default []
        feats_str: list
            List of features of type string, default []
        feats_bin: list
            List of features of type binary, default []
        feats_ord: list
            List of features of type ordinal, default []
        feats_date: list
            List of features of type date, default []
        feats_other: list
            List of features of other types, default []
        feats_cluster: list
            List of features of cluster, default []
        bin_transform: bool
            Bool indicates if want to transform the binary types to 0 and 1 
        cardinal_threshold: int
            Minumum cardinality threshold to differentiate between ordinal and continue
        symbols_nothing: list
            List of symbols which want to replace with nothing in the names of the features, default [',','.', ';', "'", '´','*', '~']
        symbols_underscore: list
            List of symbols which want to replace with underscore in the names of the features, default [' ','|', '/', '\\']
        method: str
            Method to apply to the feature's names, 'upper' to make all capital letters, 'title' to make all lowers except the first letter of each word, None to make all lower, default None
        
        """
        self.feats_id = feats_id
        self.feats_bin = feats_bin
        self.feats_cont = feats_cont
        self.feats_ord = feats_ord
        self.feats_str = feats_str
        self.feats_date = feats_date
        self.feats_other = feats_other
        self.feats_tgt = feats_tgt
        self.feats_cluster = feats_cluster
        self.__bin_transform = bin_transform        
        self.__cardinal_threshold = cardinal_threshold
        self.__symbols_nothing = symbols_nothing
        self.__symbols_underscore = symbols_underscore
        self.__method = method
        self.__all = self.feats_tgt+self.feats_id+self.feats_bin+self.feats_cont+self.feats_ord+self.feats_str+self.feats_date+self.feats_other+self.feats_cluster

   
    
    def transform(self,df):
        
        check_is_fitted(self, ['names'])
        work_df = df.copy()
        work_df.rename(columns=self.names,inplace=True)
        
        
        if self.__bin_transform:
            for col in self.feats_bin:    
                if work_df[col].dtype == bool:
                    work_df[col]=work_df[col].astype(int)
                    
                elif (work_df[col].dtype == 'object'):
                    category = [x for x in work_df[col].unique() if x==x]
                    dic = {k:category.index(k) for k in category}
                    work_df[col].replace(dic,inplace=True)
                
        work_df[self.feats_str] = work_df[self.feats_str].astype(str)
   
        return work_df
    
    def fit(self, df):
        num_types= ['i','u','f']
        date_types = ['m','M']
        str_types = ['O','S','U']
        other_types =[ 'c','V']
        
        work_df = df.copy() 
        feats = [col for col in work_df.columns if col not in self.__all ]
        self.feats_other = self.feats_other + [self.hashable(work_df[col]) for col in feats if self.hashable(work_df[col])!=None]
        feats = list(set(feats)-set(self.feats_other))
        
        cardinality = work_df[feats].nunique()
        self.feats_id = self.feats_id + [col for col in feats if col.strip()[:3].replace(' ','_').lower()=='id_' or col.strip()[-3:].replace(' ','_').lower()=='_id'  or col.strip().lower()=='id']
        self.feats_bin = self.feats_bin + [col for col in feats if cardinality.loc[col]==2]
        
        not_bin = [x for x in feats if x not in self.feats_bin+self.feats_id+self.feats_other]
        
        self.feats_str = self.feats_str + work_df[not_bin].dtypes[work_df[not_bin].dtypes.map(lambda x:x.kind in str_types)].index.to_list()
        self.feats_date = self.feats_date + work_df[not_bin].dtypes[work_df[not_bin].dtypes.map(lambda x:x.kind in date_types)].index.to_list() 
        self.feats_other = work_df[not_bin].dtypes[work_df[not_bin].dtypes.map(lambda x:x.kind in other_types)].index.to_list() + self.feats_other
        nums = work_df[not_bin].dtypes[work_df[not_bin].dtypes.map(lambda x:x.kind in num_types)].index.to_list()
        
        self.feats_cont = self.feats_cont + [col for col in nums if cardinality.loc[col]>self.__cardinal_threshold]
        self.feats_ord = self.feats_ord + [col for col in nums if col not in self.feats_cont]
        
        self.names={}
        
        if self.__bin_transform:
            self.bin_dict = {}
        for col in self.feats_bin:
            
            if self.__bin_transform:
                
                if work_df[col].dtype == bool:
                    
                    self.names.update({col:f'bin_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('bin_') else col})
                    work_df[col]=work_df[col].astype(int)
                    
                elif (work_df[col].dtype=='object'):
                    category = [x for x in work_df[col].unique() if x==x]
                    dic = {k:category.index(k) for k in category}
                    self.bin_dict.update({col:category})
                   
                    self.names.update({col:f"bin_{self.clean_names(col,symbols_nothing=self.__symbols_nothing,symbols_underscore=self.__symbols_underscore, method=self.__method)}" if not col.startswith('bin_') else col})
                    work_df[col].replace(dic,inplace=True)
                
                else:
                    self.names.update({col:f'bin_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('bin_') else col})
                
            else:
                self.names.update({col:f'bin_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('bin_') else col})
        
       
               
        self.names.update({col:self.__id_name(self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)) for col  in self.feats_id })
        self.names.update({col:f'cont_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('cont_') else col for col  in self.feats_cont  })
        self.names.update({col:f'ord_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('ord_') else col for col  in self.feats_ord  })
        self.names.update({col:f'str_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('str_') else col for col  in self.feats_str  })
        self.names.update({col:f'date_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('date_') else col for col  in self.feats_date  })
        self.names.update({col:f'other_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('other_') else col for col  in self.feats_other  })
        self.names.update({col:f'tgt_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('tgt_') else col for col  in self.feats_tgt  })
        self.names.update({col:f'cl_{self.clean_names(col,symbols_nothing=self.__symbols_nothing, symbols_underscore=self.__symbols_underscore, method=self.__method)}' if not col.startswith('cl_') else col for col  in self.feats_cluster  })
        
        self.feats_tgt = [col for col in self.names.values() if col.startswith('tgt_')]             
        self.feats_id = [col for col in self.names.values() if col.startswith('id')]
        self.feats_bin = [col for col in self.names.values() if col.startswith('bin_')]
        self.feats_cont = [col for col in self.names.values() if col.startswith('cont_')]
        self.feats_ord = [col for col in self.names.values() if col.startswith('ord_')]
        self.feats_str = [col for col in self.names.values() if col.startswith('str_')]
        self.feats_date = [col for col in self.names.values() if col.startswith('date_')]
        self.feats_other = [col for col in self.names.values() if col.startswith('other_')]
        self.feats_cluster = [col for col in self.names.values() if col.startswith('cl_')]
        self.feats_numeric = self.feats_bin+self.feats_cont+self.feats_ord
        return self
    
    def update_lists(self, df):
        """Updates lists  if there are new features with the nomenclature, if not they will go to feats_unkown
        """
        
        check_is_fitted(self, ['names'])
        work_df = df.copy()
        self.feats_tgt = [col for col in work_df.columns if col.startswith('tgt_')]             
        self.feats_id = [col for col in work_df.columns if col.startswith('id')]
        self.feats_bin = [col for col in work_df.columns if col.startswith('bin_')]
        self.feats_cont = [col for col in work_df.columns if col.startswith('cont_')]
        self.feats_ord = [col for col in work_df.columns if col.startswith('ord_')]
        self.feats_str = [col for col in work_df.columns if col.startswith('str_')]
        self.feats_date = [col for col in work_df.columns if col.startswith('date_')]
        self.feats_cluster = [col for col in work_df.columns if col.startswith('cl_')]
        self.feats_other = [col for col in work_df.columns if col.startswith('other_')]
        self.feats_numeric = self.feats_bin+self.feats_cont+self.feats_ord
        self.feats_unkown = [col for col in work_df.columns if col not in self.feats_tgt+self.feats_id
                                                                               +self.feats_bin+self.feats_cont
                                                                               +self.feats_ord+self.feats_str
                                                                               +self.feats_date+self.feats_cluster
                                                                               +self.feats_other]
        
    @staticmethod
    def __id_name(s):
        
        if s[-3:].lower() == '_id':
            s = f"id_{s[:-3]}"
            
        elif s.lower()=='id':
            s = s.lower()
            
        elif s[:3].lower() == 'id_': 
            pass
        
        else:
            s = f"id_{s}"
        return s

    @staticmethod
    def clean_names(s,symbols_nothing=[',','.', ';', "'", '´','*', '~'], symbols_underscore=[' ','|', '/', '\\'], method=None):
        s =  str(s).strip()
        s = s.title() if method=='title' else( s.upper() if  method=='upper' else s.lower())
        for x in symbols_nothing:
            s = s.replace(x,'')
        for y in symbols_underscore:   
            s = s.replace(y,'_')
        return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    
    @staticmethod
    def hashable(col):
        try:
            col.unique()
        except:
            return col.name


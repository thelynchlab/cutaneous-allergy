import pandas as pd
import numpy as np
import contact

from sklearn import preprocessing
from sklearn import naive_bayes 
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pickle
import csv
#from sklearn.naive_bayes import GaussianNB
# look at https://www.bmj.com/content/351/bmj.h3868
#todo plot adaboost: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py

BASE = '/data/contact/dec2020/'
#BASE = '/Users/magnuslynch/Work/Non Current/Research/Contact/feb2020/m
MODELS = BASE+'models/'
DATA = BASE+'processed.csv'
FIGS = BASE+'figs/'
RESULTS = BASE+'results/'
VARS = ['age_cat','date_cat','sex','site','occupation','duration','spread','atopy','psoriasis','housework','fh_atopy']
ALLERGENS = [x for x in contact.ALLERGENS if x not in contact.IGNORE]

class PrettyNames:
    def __init__(self):
        reader =  csv.reader(open('/data/contact/pretty_names.csv','r'))
        self.names = {}
        i=0
        for row in reader:
            if i>0: self.names[row[0]] = row[1]
            i+=1

    def get(self,short): return self.names.get(short,short)


def load_data():
    data = pd.read_csv(DATA)
    for a in ALLERGENS:
        data[a].mask(data[a] == -1, 'N', inplace=True)
        data[a].mask(data[a] == 1, 'Y', inplace=True)
        data[a].mask(data[a] == 0, np.NaN, inplace=True)
    
    return data 


class Transform:
    """
    Encode / decode text-based labels to numerical classes using scikit learn
    Specify whether or not to generate a new encoding
    Note: Encoding used here must match the one on which model was trained
    """
    def __init__(self):
        self.predictor_encoder = None 
        self.target_encoder = None 
   
    def create_encoders(self,data,variables=VARS):
        """
        Create the encoding scheme
        """
        self.predictor_encoder = preprocessing.OrdinalEncoder()
        self.predictor_encoder.fit(data[variables])

    def encode_predictors(self,data,vars):
        """
        @data: pandas dataframe
        @train_vars: list of column names in the dataframe to use for prediction
        @target_var: the column to be rpredicted 
        """
        X = self.predictor_encoder.transform(data[vars])
        return X

    def encode_targets(self,data,var):
        Y = data[var]
        Y = Y.replace('Y',1)
        Y = Y.replace('N',0)
        return list(Y)


def prep_data(data,target,variables=VARS):
    data=data.dropna(subset=variables)
    data=data.dropna(subset=[target])
    transform = Transform()
    transform.create_encoders(data,variables)
    X = transform.encode_predictors(data,variables)
    Y = transform.encode_targets(data,target)
    return X,Y


def get_targets(data):
    """
    Get the colnames for all targets that can be predicted
    """
    a = set(data.columns)
    b = set(VARS)
    b = b.union(set(['id','age','date_cat','age_cat','allergy_code','asthma','LYRAL','date','PPD','allergy_photosensitivity','allergy_unexplained','allergy_relevant','allergy_missed']))
    targets = sorted(list(a-b))
    print('number of targets',len(targets))
    return targets 


def create_models():
    models = {}
    models['logistic'] = LogisticRegression(solver='lbfgs',max_iter=1000)
    models['adaboost'] = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,random_state=0)
    models['random forest'] = RandomForestClassifier(random_state=0,n_estimators=100)
    models['gradient boosting'] = GradientBoostingClassifier()
    return models


def compare_classifiers_calc(data,targets=None):
    """
    Compare the performance of all the classifiers with default parameters
    using N-fold cross validation
    """
    if targets is None: 
        targets = get_targets(data)
    models = create_models()

    results = []
    
    for target in targets:
        for model_name,model in models.items():
            X,Y = prep_data(data,target)
            scores = cross_val_score(model,X,Y,cv=4,scoring='roc_auc')
            v = [target,model_name,scores.tolist()]
            results.append(v)
            print(v)

    pickle.dump(results,open(RESULTS+'compare_classifiers.pkl','wb'))


def analyse_compare_classifiers(modes=['calc_stats']):
    """
    make a table of the predictive power of the different classifiers
    """
    results = pickle.load(open(RESULTS+'compare_classifiers.pkl','rb'))
    out = open(FIGS+'analyse_compare_classifiers.csv','w') 
    order = []
    index = {}
    pretty = PrettyNames()

    if 'make_table' in modes:
        order_by = 'gradient boosting'

        for v in results:
            outcome = v[0]; model = v[1]
            mean = np.mean(v[2])
            std = np.std(v[2])
            if not outcome in index: index[outcome] = {}
            index[outcome][model] = (mean,std)
            if model == order_by: 
                order.append((mean,std,outcome))

        order.sort(reverse=True)
        out.write('outcome,model,mean,std'+'\n')
        for mean,std,outcome in order:
            outcome_p = pretty.get(outcome)
            out.write(','.join([str(x) for x in [outcome,outcome_p,order_by,mean,std]])+'\n')
            for model in sorted(index[outcome]):
                if model==order_by: continue
                out.write(','.join([str(x) for x in [outcome,outcome_p,model,index[outcome][model][0],index[outcome][model][1]]])+'\n')
        print(order)

    if 'calc_stats' in modes:
        from scipy.stats import f_oneway
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        anova_data = {}
        for x in results:
            if not x[1] in anova_data: anova_data[x[1]] = []
            anova_data[x[1]].append(np.mean(x[2]))

        for k,v in anova_data.items():
            print(k,np.mean(v),np.std(v))

        anova_data =  [list(v) for k,v in anova_data.items()]
        anova = f_oneway(*anova_data)
        print(anova)

        endog = []
        groups = []
        
        for x in results:
            endog.append(np.mean(x[2]))
            groups.append(x[1])

        tukey_hsd = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=0.05)
        print(tukey_hsd)
        import pdb; pdb.set_trace()


def plot_compare_classifiers(modes=['bar']):
    results = pickle.load(open(RESULTS+'compare_classifiers.pkl','rb'))
    pretty = PrettyNames()
    
    if 'scatter' in modes:
        #Plot scatter of all the predictions for each classifier
        v = {}
        for r in results: 
            if not r[1] in v: v[r[1]] = []
            v[r[1]].append(np.mean(r[2]))
        
        model_names = sorted(v.keys())
        model_names.remove('logistic')
        model_names.insert(0,'logistic')
        
        print(model_names)
        X = []; Y = []
        X_av = []; Y_av = []

        for key,val in v.items(): 
            i = model_names.index(key)
            for a in val:
                X.append(i)
                Y.append(a)
            X_av.append(i)
            Y_av.append(np.mean(val))

        
        print(X,Y)
        import pylab as pl
        pl.scatter(X,Y,color='black',s=3)
        pl.scatter(X_av,Y_av,color='red',marker='_')
        pl.savefig(FIGS+'compare_classifiers.eps')
        pl.show()
        pl.clf()


    #Plot the individual allergens as bar chart for each model
    if 'bar' in modes:
        v = {}
        for r in results:
            if not r[1] in v: v[r[1]] = [] 
            v[r[1]].append((r[0],np.mean(r[2]),np.std(r[2])))
        

        w = {}
        for key,val in v.items():
            val = sorted(val)
            w[key] = {}
            w[key]['names'] = [pretty.get(x[0]) for x in val]
            w[key]['mean'] = [x[1] for x in val]
            w[key]['std'] = [x[2] for x in val]
         
        import pylab as pl
        import matplotlib
        for model_name,data in w.items():
            X = np.arange(len(data['mean'])) 
            fig = pl.figure()
            ax = fig.add_axes([0.1,0.5,0.9,0.5])
            ax.bar(X,data['mean'],width=0.7,yerr=data['std'])
            matplotlib.rc('xtick', labelsize=6) 
            matplotlib.rc('ytick', labelsize=10) 
            pl.xticks(X,data['names'],rotation=90)
            #pl.show()
            filename = FIGS+'bar_'+model_name+'.eps'
            print(filename)
            pl.savefig(filename)
            pl.clf()


def plot_roc_curve(model,target,model_name,X_train,X_test,Y_train,Y_test,filename):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    test_score = model.decision_function(X_test)
    fpr_test, tpr_test, _ = roc_curve(Y_test, test_score)
    roc_auc_test = auc(fpr_test, tpr_test)
    train_score = model.decision_function(X_train)
    fpr_train, tpr_train, _ = roc_curve(Y_train, train_score)
    roc_auc_train = auc(fpr_train, tpr_train)
   
    import pylab as pl
    import matplotlib
    pl.figure()
    lw = 2
    pl.plot(fpr_test, tpr_test, color='darkorange',lw=lw, label='Validation dataset (area = %0.2f)' % roc_auc_test)
    pl.plot(fpr_train, tpr_train, color='blue',lw=lw, label='Training dataset (area = %0.2f)' % roc_auc_train)
    pl.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10) 
    pl.title('Receiver operating characteristic '+target+' '+model_name)
    pl.legend(loc="lower right")
    print(filename)
    pl.savefig(filename)
    pl.clf()


def plot_auc(data):
    """
    For each logistic and GBM plot AUC with a random training and test dataset
    compare AUC with all data versus only training data
    """
    targets = get_targets(data)
    models = create_models()    
    
    results = []
    
    for target in targets:
        for model_name,model in models.items():
            if model_name not in ['logistic','gradient boosting']: continue
            X,Y = prep_data(data,target)
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
            model.fit(X_train,Y_train)
            Y_pred = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(Y_test,Y_pred)
            print(target,model_name,auc)
            plot_roc_curve(model,target,model_name,X_train,X_test,Y_train,Y_test,FIGS+'new_auc/'+target+'_'+model_name+'.eps')



def stats(data):
    print('age',np.mean(data['age']),np.std(data['age']))
    print('atopy',np.sum(data['atopy']=='Y') / data['atopy'].count())
    print('spread',np.sum(data['spread']=='Y') / data['spread'].count())
    print('male',np.sum(data['sex']=='M') / len(data))
    print('female',np.sum(data['sex']=='F') / len(data))
    print('housework full time',np.sum(data['housework']=='full_time') / data['housework'].count())
    print('housework part time',np.sum(data['housework']=='part_time') / data['housework'].count())
    print('housework some',np.sum(data['housework']=='some_regularity') / data['housework'].count())
    print('housework no',np.sum(data['housework']=='no_housework') / data['housework'].count())
    import pdb; pdb.set_trace()


def interpret_gradient_boosting(data):
    """
    Train with interpretable gradient boosting
    Use 4-fold cross validation and compute mean and s.d. for each of the interpreted parameters
    Output: table of importance values for each allergen
    """
    out = open(FIGS+'interpret_gradient_boosting.csv','w') 
    header = ['Target']
    header.extend(VARS)
    out.write(','.join(header)+'\n')
    
    pretty = PrettyNames()
    targets = get_targets(data)
    models = create_models()
    
    for target in targets: 
        results = {}
        X,Y = prep_data(data,target)
        model = models['gradient boosting']
        model.fit(X,Y)
        row = ['"%s"'%pretty.get(target)]
        row.extend(['%.2f' % x for x in model.feature_importances_])
        print(row)
        out.write(','.join(row)+'\n')


def calibrate_classifiers_brier(data):
    """
    Brier score loss: https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py
    """
    out = open(FIGS+'calibrate_classifiers_brier.csv','w') 
    out.write('Target,Model,Mean,Standard deviation\n')
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=4)
    
    models = create_models()
    targets = get_targets(data)
    pretty = PrettyNames()

    for target in targets:
        #target = 'NICKEL'
        X,Y = prep_data(data,target)
        Y = np.array(Y)

        for model_name in models:
            vals = []
            for train_index,test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                #X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
                model = models[model_name]
                model.fit(X_train,Y_train)
                pred = model.predict_proba(X_test)[:,1]
                loss = brier_score_loss(Y_test, pred)
                vals.append(loss)

            mean = np.mean(vals); std = np.std(vals)
            row = ['"%s"'%pretty.get(target),model_name,'%.3f'%mean,'%.3f'%std]
            out.write(','.join(row)+'\n')
            print(row)



def make_calibrated_classifiers(data):
    """
    Generate calibrated classifiers for allergy_now and one_pos for logistic and gradient boosting
    """
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibratedClassifierCV
    import pylab as pl

    models = create_models()



    for target in ['one_pos','allergy_now']:
        X,Y = prep_data(data,target)
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)
        Y = np.array(Y)
        max_x = 0.0
        max_y = 0.0
        
        for model_name in ['gradient boosting','logistic']:
            model = models[model_name]
            model.fit(X_train,Y_train)
            calibrated_model = CalibratedClassifierCV(base_estimator=model, cv=3,method='isotonic')
            calibrated_model.fit(X_train,Y_train)
            pl.clf()
            nbins = 5
            model_pred = model.predict_proba(X_test)[:,1]
            pl.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
            calibrated_pred = calibrated_model.predict_proba(X_test)[:,1]
            fraction_of_positives,mean_predicted_value = calibration_curve(Y_test,model_pred,n_bins=nbins)
            pl.plot(mean_predicted_value,fraction_of_positives,label='uncalibrated')
            fraction_of_positives,mean_predicted_value = calibration_curve(Y_test,calibrated_pred,n_bins=nbins)
            pl.plot(mean_predicted_value,fraction_of_positives,label='calibrated')
            pl.legend()
            pl.xlim([0,1])
            pl.ylim([0,1])
            pl.xlabel('Mean predicted value')
            pl.ylabel('Fraction of positives')
            #pl.show()
            
            fig_name = FIGS+'make_calibrated_classifiers/'+model_name+'_'+target+'.eps'
            print(fig_name)
            print(mean_predicted_value)
            print(fraction_of_positives)
            pl.savefig(fig_name)
            
            model_name = FIGS+'make_calibrated_classifiers/'+model_name+'_'+target+'.pkl'
            pickle.dump(calibrated_model,open(model_name,'wb'))




def logistic_remove_variables(data):
    """
    Test removing variables from logistic regression with lower p value
    """
    from sklearn.model_selection import KFold
    sig_vars = {'allergy_now':['age_cat','date_cat','sex','site','occupation','duration','spread','atopy'],'one_pos': ['age_cat','date_cat','sex','site','occupation','duration','spread','atopy','housework']}

    model = create_models()['logistic']
    kf = KFold(n_splits=4)

    results = {}

    for target,variables in sig_vars.items():
        if not 'target' in results: results[target]={}
        
        #only significant variables
        X,Y = prep_data(data,target,variables)
        Y = np.array(Y)

        for train_index,test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
             
            model.fit(X_train,Y_train)
            Y_pred = model.predict_proba(X_test)[:,1]
            auc_sig = roc_auc_score(Y_test,Y_pred)
            if not 'sig' in results[target]: results[target]['sig'] = []
            results[target]['sig'].append(auc_sig)

        
        #all variables
        X,Y = prep_data(data,target,VARS)
        Y = np.array(Y)

        for train_index,test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            model.fit(X_train,Y_train)
            Y_pred = model.predict_proba(X_test)[:,1]
            auc_all = roc_auc_score(Y_test,Y_pred)
            if not 'all' in results[target]: results[target]['all'] = []
            results[target]['all'].append(auc_all)
    
    
    #calc stats + plot bar chart
    means = []
    stds = []
    labels = []
    for target in sig_vars:
        for x in ['sig','all']:
            mean = np.mean(results[target][x])
            std = np.std(results[target][x])
            means.append(mean)
            stds.append(std)
            labels.append(target+'_'+x)

    print(labels)
    print(means)
    print(stds)

    #calc stats  
    from scipy.stats import ttest_ind
    print('one_pos')
    print(ttest_ind(results['one_pos']['all'],results['one_pos']['sig']))
    print('allergy_now')
    print(ttest_ind(results['allergy_now']['all'],results['allergy_now']['sig']))

    #plot bar chart 
    import pylab as pl
    print(means,stds,labels)
    pl.bar(labels,means,yerr=stds)
    pl.savefig(FIGS+'logistic_remove_variables.eps')

    
def split_date():    
    """
    Split according to date of testing 
    Calculate AUC, plot ROC curves and calibration curves
    """ 
    data = load_data()
    models = create_models()    
    for model_name in ['logistic','gradient boosting']:

        model = models[model_name]

        split_year = 2010
        old = data[data['date']<split_year]
        new = data[data['date']>=split_year]
        print(len(old),len(new))
        
        for target in ['one_pos','allergy_now']:
            X_old,Y_old = prep_data(old,target) 
            X_new,Y_new = prep_data(new,target) 

            model.fit(X_old,Y_old)
            Y_pred = model.predict_proba(X_new)[:,1]
            auc = roc_auc_score(Y_new,Y_pred)
            print(model_name,target,auc)

            plot_roc_curve(model,target,model_name,X_old,X_new,Y_old,Y_new,FIGS+'split_date_'+target+'_'+model_name+'.eps')
    



def  main():
    data = load_data()
    data = data.sample(frac=1) #shuffle rows random order
    #calibrate_classifiers_brier(data)
    #calibrate_classifiers_plot(data)
    #make_calibrated_classifiers(data)
    #analyse_compare_classifiers(modes='calc_stats')
    #logistic_remove_variables(data)
    #test_load_model()
    #interpret_gradient_boosting(data)
    #stats(data)
    #compare_classifiers_calc(data)
    #analyse_compare_classifiers(modes=['make_table'])
    #plot_compare_classifiers(modes=['bar','scatter'])
    #plot_auc(data)
    #get_targets(data)
    #split_date()





main()




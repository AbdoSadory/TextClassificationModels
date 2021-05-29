import pandas as pd #For Data Manipulation
import re #For Data De-Noise
from sklearn.feature_extraction.text import CountVectorizer #For Text Converting
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import json 

class IMDBReviews:
    def __init__(self):
        self.cv     = CountVectorizer()
        self.le     = LabelEncoder()
        self.lr     = LinearRegression()
        self.lo     = LogisticRegression()
        self.knn    = KNeighborsClassifier(n_neighbors=7)
        self.nb     = GaussianNB()
        self.svm    = SVC(kernel="linear",random_state=0)
        self.svmP   = SVC(kernel="poly",random_state=0)
        self.svmR   = SVC(kernel="rbf",random_state=0)
        self.sw     = open("en_stopwords.txt",encoding='utf-8').read().split('\n') 
        self.res    = {"1":"Positive","0":"Negative"}

    
    def __DataSet(self):
        self.data   = pd.read_csv('IMDB Dataset5.csv')
    
    def __Data_Clean(self,text):
        text          = re.sub("[^A-Za-z0-9']"," ",text)
        words         = text.lower().split()
        wanted_words  =[]
        for word in words:
            if word not in self.sw:
                wanted_words.append(word)
        return ' '.join(wanted_words)
    
    def pre_processing(self):
        self.__DataSet()
        self.data["Cleaned_Reviews"]=self.data.review.apply(self.__Data_Clean)
        self.x = self.cv.fit_transform(self.data.Cleaned_Reviews).toarray()
        self.y = self.le.fit_transform(self.data.sentiment)
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size = 0.2,random_state=0)
#---------------------------------------------------{Logistic Regression}-------------------------------------------------------
    def LoModelFitting(self):
            self.pre_processing()
            self.lo.fit(self.x_train,self.y_train)  
    def LoEvaluate(self):
            if len(self.lo.classes_)>0:
                self.Test_Scorelo= self.lo.score(self.x_test,self.y_test)
                return self.Test_Scorelo
            else:
                self.LoModelFitting()
                self.LoEvaluate()       
    def Lopredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.lo.predict(text)[0])
            return self.res[res]
#---------------------------------------------------{Linear Regression}---------------------------------------------------------    
    def LRModelFitting(self):
            self.pre_processing()
            self.lr.fit(self.x_train,self.y_train)
    def LREvaluate(self):
                self.Test_Scorelr= self.lr.score(self.x_test,self.y_test)
                return self.Test_Scorelr
    def LRpredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.lr.predict(text)[0])
            return self.res[res]
#---------------------------------------------------------{KNN}----------------------------------------------------------------    
    def KNNModelFitting(self):
            self.pre_processing()
            self.knn.fit(self.x_train,self.y_train)
    def KNNEvaluate(self):
            if len(self.knn.classes_)>0:
                self.Test_Scoreknn= self.knn.score(self.x_test,self.y_test)
                return self.Test_Scoreknn
            else:
                self.KNNModelFitting()
                self.EvaluateKNN()
    def KNNpredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.knn.predict(text)[0])
            return self.res[res]
#-----------------------------------------------------{Naive Bayes}-------------------------------------------------------------
    def NBModelFitting(self):
            self.pre_processing()
            self.nb.fit(self.x_train,self.y_train)
            
    def NBEvaluate(self):
            if len(self.nb.classes_)>0:
                self.Test_ScoreNB= self.nb.score(self.x_test,self.y_test)
                return self.Test_ScoreNB
            else:
                self.NBModelFitting()
                self.EvaluateNB()
        
    def NBpredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.nb.predict(text)[0])
            return self.res[res]
#-------------------------------------------------------{SVM}---------------------------------------------------------------------
    def SVMModelFitting(self):
            self.pre_processing()
            self.svm.fit(self.x_train,self.y_train)
            
    def SVMEvaluate(self):
            if len(self.svm.classes_)>0:
                self.SVMTest_Score= self.svm.score(self.x_test,self.y_test)
                return self.SVMTest_Score
            else:
                self.SVMModelFitting()
                self.SVMEvaluate()
        
    def SVMpredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.svm.predict(text)[0])
            return self.res[res]
#--------------------------------------------------{SVM-Polynomial}-------------------------------------------------------------
    def SVMpModelFitting(self):
            self.pre_processing()
            self.svmP.fit(self.x_train,self.y_train)
            
    def SVMpEvaluate(self):
            if len(self.svmP.classes_)>0:
                self.SVMpTest_Score= self.svmP.score(self.x_test,self.y_test)
                return self.SVMpTest_Score
            else:
                self.SVMpModelFitting()
                self.SVMpEvaluate()
        
    def SVMppredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.svmP.predict(text)[0])
            return self.res[res]
#-----------------------------------------------------{SVM-rbf}-----------------------------------------------------------------
    def SVMrModelFitting(self):
            self.pre_processing()
            self.svmR.fit(self.x_train,self.y_train)
            
    def SVMrEvaluate(self):
            if len(self.svmR.classes_)>0:
                self.SVMrTest_Score= self.svmR.score(self.x_test,self.y_test)
                return self.SVMrTest_Score
            else:
                self.SVMrModelFitting()
                self.SVMrEvaluate()
        
    def SVMrpredict(self,text):
            text      = self.__Data_Clean(text)
            text      = self.cv.transform([text]).toarray()
            res       = str(self.svmR.predict(text)[0])
            return self.res[res]
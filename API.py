from flask import Flask,request,jsonify
from PythonCodeMLModel import IMDBReviews
app=Flask(__name__)
IM = IMDBReviews()
@app.route("/")
def home():
    return render_template('WebHTMLAPP.html')
#---------------------------------------------------{Logistic Regression}-------------------------------------------------------
@app.route('/LoModelFitting')
def ModelFitting1():
    IM.LoModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data) #The same page with different value !
@app.route('/LoEvaluate') #The previous link, in order!
def Evaluate1():
    IM.LoEvaluate()
    data = {"status_code":200,"accuracy":IM.LoEvaluate()}
    return jsonify(data) #The same page with different value !
@app.route('/Lopredict') #The previous link, in order!
def predict1():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.Lopredict(text)}
    return jsonify(data)
#---------------------------------------------------{Linear Regression}--------------------------------------------------------
@app.route('/LRModelFitting')
def ModelFitting2():
    IM.LRModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/LREvaluate')
def Evaluate2():
    data = {"status_code":200,"accuracy":IM.LREvaluate()}
    return jsonify(data)
@app.route('/LRpredict')
def predict2():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.LRpredict(text)}
    return jsonify(data)
#---------------------------------------------------------{KNN}----------------------------------------------------------------
@app.route('/KNNModelFitting')
def ModelFitting3():
    IM.KNNModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/KNNEvaluate')
def Evaluate3():
    data = {"status_code":200,"accuracy":IM.KNNEvaluate()}
    return jsonify(data)
@app.route('/KNNpredict')
def predict3():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.KNNpredict(text)}
    return jsonify(data)
#-----------------------------------------------------{Naive Bayes}-----------------------------------------------------------
@app.route('/NBModelFitting')
def ModelFitting4():
    IM.NBModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/NBEvaluate')
def Evaluate4():
    data = {"status_code":200,"accuracy":IM.NBEvaluate()}
    return jsonify(data)
@app.route('/NBpredict')
def predict4():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.NBpredict(text)}
    return jsonify(data)
#-------------------------------------------------------{SVM}------------------------------------------------------------------
@app.route('/SVMModelFitting')
def ModelFitting5():
    IM.SVMModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/SVMEvaluate')
def Evaluate5():
    data = {"status_code":200,"accuracy":IM.SVMEvaluate()}
    return jsonify(data)
@app.route('/SVMpredict')
def predict5():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.SVMpredict(text)}
    return jsonify(data)
#--------------------------------------------------{SVM-Polynomial}-----------------------------------------------------------
@app.route('/SVMpModelFitting')
def ModelFitting6():
    IM.SVMpModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/SVMpEvaluate')
def Evaluate6():
    data = {"status_code":200,"accuracy":IM.SVMpEvaluate()}
    return jsonify(data)
@app.route('/SVMppredict')
def predict6():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.SVMppredict(text)}
    return jsonify(data)
#-----------------------------------------------------{SVM-rbf}---------------------------------------------------------------
@app.route('/SVMrModelFitting')
def ModelFitting7():
    IM.SVMrModelFitting()
    data = {"status_code":200,"message":"Model Trained Successfully"}
    return jsonify(data)
@app.route('/SVMrEvaluate')
def Evaluate7():
    data = {"status_code":200,"accuracy":IM.SVMrEvaluate()}
    return jsonify(data)
@app.route('/SVMrpredict')
def predict7():
    text = request.args['text']
    data = {"status_code":200,'text':text,"accuracy":IM.SVMrpredict(text)}
    return jsonify(data)
app.run(debug=True)
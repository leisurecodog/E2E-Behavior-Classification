import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def Oversampling(x, y, m='SMOTE'):
    oversample = ''
    if m == 'SMOTE':
        from imblearn.over_sampling import SMOTE
        oversample = SMOTE()
    return oversample.fit_resample(x, y)

def Undersampling(x, y, m='ENN'):
    from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
    undersample = ''
    if m == 'ENN':
        undersample = EditedNearestNeighbours()
    elif m == 'Tomek':
        undersample = TomekLinks()
        
    return undersample.fit_resample(x, y)

def show_metrics(y_total, pred_total, flag=False):
    from sklearn.metrics import confusion_matrix, classification_report
    target_names = ['conservative', 'aggressive']
    if isinstance(y_total[0], np.ndarray):
        y_total = [np.argmax(y) for y in y_total]
        pred_total = [np.argmax(y) for y in pred_total]
    # print("y GT is: {}".format(y_total))
    # print("y pred is: {}".format(pred_total))

    import warnings
    warnings.filterwarnings("error")

    final = classification_report(y_total, pred_total, labels=[0,1], target_names=target_names, output_dict=True, zero_division=1)
    if flag:
        final = classification_report(y_total, pred_total, labels=[0,1], target_names=target_names, zero_division=1)
        print(final)
    return final



def pytorch_test(test_dataloader, opt):
    from model import Model
    from FLoss import FocalLoss, focal_loss, BinaryFocalLossWithLogits
    import torch.optim as optim
    import torch.utils.data as Data
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(opt.id_num)
    model.load_state_dict(torch.load('./test_model.pth'))
    model.eval()
    model.to(device)
    pred_total = []
    y_total = []
    for x, y in test_dataloader:
        
        x, y = x.to(device=device), y.to(device=device)
        x = x.unsqueeze(1)
        pred = model(x)
        pred = pred.cpu().detach().numpy()
        y_total.append(y.cpu().detach().numpy())
        pred_total.extend(list(pred))

    show_metrics(y_total, pred_total, True)

def pytorch_train(new_embedding, labels, opt):
    from model import Model
    from FLoss import FocalLoss, focal_loss, BinaryFocalLossWithLogits
    import torch.optim as optim
    import torch.utils.data as Data
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print(device)
    Xtrain, Xtest, ytrain, ytest = train_test_split(new_embedding, labels, test_size=0.1)

    # print(np.shape(Xtrain), np.shape(ytrain))
    if opt.oversampling:
        Xtrain, ytrain = Oversampling(Xtrain, ytrain)
    Xtrain = torch.Tensor(Xtrain)
    Xtest = torch.Tensor(Xtest)

    labels_one_hot_train = np.eye(2)[ytrain]
    ytrain = torch.Tensor(labels_one_hot_train)

    labels_one_hot_test = np.eye(2)[ytest]
    ytest = torch.Tensor(labels_one_hot_test)

    torch_dataset = Data.TensorDataset(Xtrain, ytrain)
    train_dataloader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = opt.batch_size,
        shuffle = True,
        num_workers = opt.worker,
    )
    torch_dataset_test = Data.TensorDataset(Xtest, ytest)
    test_dataloader = Data.DataLoader(
        dataset = torch_dataset_test,
        batch_size = 1,
        shuffle = True,
        num_workers = opt.worker,
    )

    model = Model(opt.id_num)
    model.to(device)

    criterion = torch.nn.BCELoss()
    if opt.focal_loss:
        criterion = BinaryFocalLossWithLogits()
    
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    for epoch in range(opt.epochs):
        total_loss = 0.0
        for x, y in train_dataloader:
            x, y = x.to(device=device), y.to(device=device)
            x = x.unsqueeze(1)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % opt.train_log == 0:
            print("epoch {}/{}\t: loss: {}".format(epoch, opt.epochs, total_loss))
    # torch.save(model.state_dict(), './test_model.pth')
    if opt.test:
        pytorch_test(test_dataloader, opt)

def bayes_train_test(xtrain, ytrain, xtest, ytest):

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(xtrain, ytrain).predict(xtest)
    show_metrics(ytest, y_pred)

def svm_train_test(xtrain, ytrain, xtest, ytest):
    model = svm.SVC(max_iter=1000)
    y_pred = model.fit(xtrain, ytrain).predict(xtest)
    show_metrics(ytest, y_pred)

def xgboost_train_test(xtrain, ytrain, xtest, ytest):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    # n_estimators = [50,100,150,200,250,300]
    # max_depth = [1,2,3]
    # learning_rate=[0.001,0.005,0.01,0.015,0.02,0.03,0.04,0.05]
    # scale_pos_weight = [1,2,3,4,5,6,7,8,9,10]
    # params = {'n_estimators': n_estimators,
    #             'max_depth': max_depth,
    #             'learning_rate': learning_rate,
    #             'scale_pos_weight': scale_pos_weight}
    # model = XGBClassifier()
    model = XGBClassifier(learning_rate=0.05, max_depth=3, n_estimators=250, scale_pos_weight=1, use_label_encoder=False)
    # gscv = GridSearchCV(model, params, cv=10, verbose=2, scoring='neg_mean_squared_error', n_jobs=-1)

    # gscv.fit(xtrain, ytrain)
    # print(gscv.best_params_)
    y_pred = model.fit(xtrain, ytrain).predict(xtest)
    return show_metrics(ytest, y_pred)

def rf_train_test(xtrain, ytrain, xtest, ytest):
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    y_pred = forest.fit(xtrain, ytrain).predict(xtest)
    return show_metrics(ytest, y_pred)

def imb_xgboost_train_test(xtrain, ytrain, xtest, ytest):
    print(np.array(xtrain).shape, np.array(ytrain).shape)
    from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
    from sklearn.model_selection import GridSearchCV
    xgboster_focal = imb_xgb(special_objective='focal')
    xgboster_weight = imb_xgb(special_objective='weighted')
    CV_focal_booster = GridSearchCV(xgboster_focal, {"focal_gamma":[0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0,2.5,3.0]})
    CV_weight_booster = GridSearchCV(xgboster_weight, {"imbalance_alpha":[0,0.1,0.2,0.5,0.8,1.0,1.5,2.0,2.5,3.0,4.0,4.5,5.0,5.5,6.0]})
    CV_focal_booster.fit(np.array(xtrain), np.array(ytrain))
    CV_weight_booster.fit(np.array(xtrain), np.array(ytrain))
    opt_focal_booster = CV_focal_booster.best_estimator_
    opt_weight_booster = CV_weight_booster.best_estimator_ 
    y_pred = opt_focal_booster.predict_two_class(np.array(xtest), y=None) 
    res1 = show_metrics(ytest, [np.argmax(l) for l in y_pred])
    y_pred = opt_weight_booster.predict_two_class(np.array(xtest), y=None) 
    res2 = show_metrics(ytest, [np.argmax(l) for l in y_pred])
    return res1, res2

def osvm_train_test(xtrain, ytrain, xtest, ytest):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import OneClassSVM
    
    training_conservative = []
    training_aggressive = []
    conservative_remain_data = {}
    # training convservative data
    model1 = OneClassSVM()
    for idx, y in enumerate(ytrain):
        if y == 0:
            training_conservative.append(xtrain[idx])

    model1.fit(training_conservative)
    yhat1 = model1.predict(xtest)
    
    for i in range(len(yhat1)):
        if yhat1[i] == 1:
            conservative_remain_data[i] = yhat1[i]
            yhat1[i] = 0
        elif yhat1[i] == -1:
            yhat1[i] = 1

    # training aggressive data
    model2 = OneClassSVM(nu=0.8)
    for idx, y in enumerate(ytrain):
        if y == 1:
            training_aggressive.append(xtrain[idx])
    model2.fit(training_aggressive)

    yhat2 = model2.predict(xtest)
    
    for i in range(len(yhat2)):
        if yhat2[i] == 1:
            yhat2[i] = 1
            if i in conservative_remain_data:
                yhat1[i] = 1
        elif yhat2[i] == -1:
            yhat2[i] = 0

    r1 = show_metrics(ytest, yhat1)

    return r1


def isolation_forest_train_test(xtrain, ytrain, xtest, ytest):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, random_state=42)
    # CV_IF_booster = GridSearchCV(clf, {"n_estimators":[50,100,150,200,250,300,350,400,450,500,550,600]}, scoring="neg_root_mean_squared_error")
    # CV_IF_booster.fit(xtrain, ytrain)
    # clf = IsolationForest(n_estimators=CV_IF_booster.best_params_['n_estimators'], random_state=42)
    clf.fit(xtrain)
    clf_pred = clf.predict(xtest)

    for i in range(len(clf_pred)):
        if clf_pred[i] == 1:
            clf_pred[i] = 0
        elif clf_pred[i] == -1:
            clf_pred[i] = 1
    # print(clf_pred)
    res = show_metrics(clf_pred, ytest)
    return res
def SUOD_train_test(xtrain, ytrain, xtest, ytest):
    from pyod.models.suod import SUOD
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    from pyod.models.copod import COPOD

    # detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
    #                  LOF(n_neighbors=25), LOF(n_neighbors=35),
    #                  COPOD(), IForest(n_estimators=100),
    #                  IForest(n_estimators=200)]

    # # decide the number of parallel process, and the combination method
    # clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average', verbose=False)

    # or to use the default detectors
    clf = SUOD(n_jobs=2, combination='average',
               verbose=False)
    clf.fit(xtrain)
    y_pred = clf.predict(xtest)
    show_metrics(y_pred, ytest, True)

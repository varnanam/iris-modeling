import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,mean_squared_error,r2_score



def main():
    
    json_file="/Users/va/Desktop/Internshala/algoparams_from_ui.json"
    print("Reading JSON File:"+ json_file)
    
    data_file="/Users/va/Desktop/Internshala/iris.csv"
    print("Data File:"+ data_file)
      
    json_file_val=open(json_file)
    json_file_dict=json.load(json_file_val)

    # Find Prediction Type    
    prediction_type=json_file_dict['design_state_data']['target']['prediction_type']
    
    print("Prediction Type: "+prediction_type)
    
    # Find Target Column    
    target_column=json_file_dict['design_state_data']['target']['target']
    
    print("Target Column: "+target_column)

    new_df=impute_features(data_file,json_file_dict)    
    print(new_df.head(4))
    
    model_data(prediction_type,target_column,json_file_dict,new_df)
    
    json_file_val.close()



def model_data(prediction_type,target_column,json_file_dict,new_df):
    # Handling Outliers
    new_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].boxplot()
    plt.show()
    cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    for n in cols:
        iqr=new_df[n].quantile(.75)-new_df[n].quantile(.25)
        up=new_df[n].quantile(.75) + 1.5*iqr
        low=new_df[n].quantile(.25) - 1.5*iqr
        outliers=new_df[(new_df[n]<low)|(new_df[n]>up)]
        #using statistics will cap outliers>up to upper limit and 
        #outliers<low to lower limit
        new_df.loc[new_df[n]<low,n]=low
        new_df.loc[new_df[n]>up,n]=up
        new_df[cols].boxplot()
        plt.show()
      
    # scaling the data
    #from sklearn.preprocessing import StandardScaler()
    #scaler=StandardScaler()
    #new_df[new_df.columns]=scaler.fit_transform(new_df)
    #new_df.describe()
    #split dataset into features and target variable
    X = new_df.drop(['petal_width'],axis=1) # Features
    y= new_df['petal_width'] # Target variable
    #Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # For Modeling
    alogrithms=json_file_dict['design_state_data']['algorithms']
    #print(alogrithms)
    
    for i in alogrithms:
       #print("Alogrithm Name: "+i)
       #print("Model Names: " + str(json_file_dict['design_state_data']['algorithms'][i]))
            
        model=['RandomForestRegressor','GBTRegressor','LinearRegression',
               'RidgeRegression','LassoRegression','ElasticNetRegression',
               'xg_boost','DecisionTreeRegressor','SGD','KNN','extra_random_trees']  
        for i in model:   
            if (i == "LinearRegression"):
                min_iter=json_file_dict['design_state_data']['algorithms'][i]['min_iter']
                max_iter=json_file_dict['design_state_data']['algorithms'][i]['max_iter']
                min_regparam=json_file_dict['design_state_data']['algorithms'][i]['min_regparam']
                max_regparam=json_file_dict['design_state_data']['algorithms'][i]['max_regparam']
                min_elasticnet=json_file_dict['design_state_data']['algorithms'][i]['min_elasticnet']
                max_elasticnet=json_file_dict['design_state_data']['algorithms'][i]['max_elasticnet']
            
                from sklearn.linear_model import LinearRegression
                lr=LinearRegression()
                lr_model=lr.fit(X_train,y_train)
                y_pred=lr_model.predict(X_test)
                
                print('LinearRegression : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',lr_model.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
                
            elif(i=="RandomForestRegressor"):
                min_trees= json_file_dict['design_state_data']['algorithms'][i]['min_trees']
                max_trees= json_file_dict['design_state_data']['algorithms'][i]['max_trees']
                feature_sampling_statergy= json_file_dict['design_state_data']['algorithms'][i]['feature_sampling_statergy']
                min_depth= json_file_dict['design_state_data']['algorithms'][i]['min_depth']
                max_depth= json_file_dict['design_state_data']['algorithms'][i]['max_depth']
                min_samples_per_leaf_min_value= json_file_dict['design_state_data']['algorithms'][i]['min_samples_per_leaf_min_value']
                min_samples_per_leaf_max_value= json_file_dict['design_state_data']['algorithms'][i]['min_samples_per_leaf_max_value']
                parallelism= json_file_dict['design_state_data']['algorithms'][i]['parallelism']

                from sklearn.ensemble import RandomForestRegressor
                rf=RandomForestRegressor(n_estimators=max_trees,max_depth=max_depth,min_samples_leaf=min_samples_per_leaf_min_value,)
                rf=rf.fit(X_train,y_train)
                y_pred=rf.predict(X_test)
                
                print('RandomForestRegressor : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',rf.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
                
            elif(i=='GBTRegressor'):
               max_depth=json_file_dict['design_state_data']['algorithms'][i]['max_depth']

               from sklearn.ensemble import GradientBoostingRegressor
               gb=GradientBoostingRegressor(max_depth=max_depth)
               gb=gb.fit(X_train,y_train)
               y_pred=gb.predict(X_test) 
            
               print('GBTRegressor : ')
               print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
               print('R2 score ',gb.score(X_test,y_test))
               #print(classification_report(y_test, y_pred))  
               print('\n')
               
            elif(i=="RidgeRegression"):
               max_iter= json_file_dict['design_state_data']['algorithms'][i]['max_iter']

               from sklearn.linear_model import Ridge
               from sklearn.pipeline import make_pipeline

               pipeline = make_pipeline(Ridge(alpha=1.0,max_iter=max_iter))
               pipeline.fit(X_train, y_train)

               y_train_pred = pipeline.predict(X_train)
               y_test_pred = pipeline.predict(X_test)

               print('RidgeRegression : ')
               print('MSE test: %.3f' %(mean_squared_error(y_test, y_test_pred)))
               print('R^2 test: %.3f' % (r2_score(y_test, y_test_pred)))
               #print(classification_report(y_test, y_test_pred))
               print('\n')
               
            elif(i=="LassoRegression"):
               max_iter= json_file_dict['design_state_data']['algorithms'][i]['max_iter']
               
               from sklearn.linear_model import Lasso

               lasso = Lasso(alpha=1.0,max_iter=max_iter)
               lasso.fit(X_train, y_train)
               y_pred=np.round(np.clip(lasso.predict(X_test),1,10)).astype(int)
            
               print('LassoRegression : ')
               print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
               print('R2 score ',lasso.score(X_test,y_test))
               #print(classification_report(y_test, y_pred))
               print('\n')
               
            elif(i=="ElasticNetRegression"):
                max_iter= json_file_dict['design_state_data']['algorithms'][i]['max_iter']
                
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=1.0, l1_ratio=0.5,max_iter=max_iter)
                model.fit(X_train, y_train)
                y_pred=model.predict(X_test)
                
                print('ElasticNetRegression : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',model.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
                
            elif(i=="xg_boost"):
                max_depth_of_tree= json_file_dict['design_state_data']['algorithms'][i]['max_depth_of_tree']
                gamma= json_file_dict['design_state_data']['algorithms'][i]['gamma']
                  
                import xgboost as xgb
                from xgboost import XGBRegressor
                model = XGBRegressor(max_depth=max_depth_of_tree[0],gamma=gamma[0])
                model.fit(X_train, y_train)
                y_pred=model.predict(X_test)
                
                print('xg_boost : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',model.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
                
            elif(i=="DecisionTreeRegressor"):
                max_depth= json_file_dict['design_state_data']['algorithms'][i]['max_depth']

                from sklearn.tree import DecisionTreeRegressor
                tree=DecisionTreeRegressor(max_depth=max_depth)
                tree.fit(X_train, y_train)
                y_pred=tree.predict(X_test)
                
                print('DecisionTreeRegressor : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',tree.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
                
            elif(i=="SGD"):
                from sklearn.linear_model import SGDRegressor
                sgd=SGDRegressor(max_iter=1000, tol=.001)
                sgd.fit(X_train, y_train)
                y_pred=sgd.predict(X_test)        
                
                print('Stochastic Gradient Descent : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',sgd.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')         
                
            elif(i=="KNN"):
                n_neighbor=json_file_dict['design_state_data']['algorithms'][i]['k_value']
                from sklearn.neighbors import KNeighborsRegressor
                knn_model = KNeighborsRegressor(n_neighbors=n_neighbor[0]).fit(X_train, y_train)
                preds = knn_model.predict(X_test)
                
                print('KNN : ')
                print('Root mean squared error is ',mean_squared_error(y_test,preds))
                print('R2 score ',knn_model.score(X_test,y_test))
                #print(classification_report(y_test, preds))
                print('\n')
                
            elif(i=='extra_random_trees'):
                n_estimator=json_file_dict['design_state_data']['algorithms'][i]['num_of_trees']
                max_depth=json_file_dict['design_state_data']['algorithms'][i]['max_depth']
                min_samples_per_leaf=json_file_dict['design_state_data']['algorithms'][i]['min_samples_per_leaf']
                from sklearn.ensemble import ExtraTreesRegressor
                etr=ExtraTreesRegressor(n_estimators=n_estimator[1],min_samples_leaf=min_samples_per_leaf[0],max_depth=max_depth[1],random_state=0).fit(X_train, y_train)
                y_pred=etr.predict(X_test) 
               
                print('extra_random_trees : ')
                print('Root mean squared error is ',mean_squared_error(y_test,y_pred))
                print('R2 score ',etr.score(X_test,y_test))
                #print(classification_report(y_test, y_pred))
                print('\n')
            else:
                print("No modeling",'\n')

def impute_features(data_file,json_file_dict):
    # Read the Iris CSV File
    # Impute Missing values based on the Alogrithm JSON
    data=pd.read_csv(data_file)
    #print(data.head(2))

    feature_details=json_file_dict['design_state_data']['feature_handling']

    for i in feature_details:
        print("\n\nDetails for feature:" + i)
        print("*****Start****")
        print("Feature Name: "+json_file_dict['design_state_data']['feature_handling'][i]['feature_name'])
        # Feature Handling for i
        if (json_file_dict['design_state_data']['feature_handling'][i]['feature_variable_type'] == "numerical"):
            if (json_file_dict['design_state_data']['feature_handling'][i]['feature_details']['impute_with'] == "Average of values"):
                print("Imputing with average values for " + i)
                data[i].fillna(data[i].mean(),inplace=True)
                #print(data.head(4))
            else:
                print("Imputing with Custom for " + i)
                data[i].interpolate(method="linear",axis=0)
                #print(data.head(4))
        elif (json_file_dict['design_state_data']['feature_handling'][i]['feature_variable_type'] == "text"):
            print("Text Handling: "+json_file_dict['design_state_data']['feature_handling'][i]['feature_details']['text_handling'])
            print("Hash Columns: "+str(json_file_dict['design_state_data']['feature_handling'][i]['feature_details']['hash_columns']))
            #print (word_tokenize(data[i])
            classes={"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3}
            data[i]=data[i].map(classes)
            #print(data[i])
        
        print("*****End****\n\n\n\n")
    return data

if __name__ == "__main__":
    main()
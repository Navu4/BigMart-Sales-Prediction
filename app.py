from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

#Load pickel file
model = pickle.load(open('model.pkl', 'rb'))



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    Item_Identifier = request.form['Item ID']
    Item_Weight = float(request.form['Weight'])

    item_fat_content=request.form['Item Fat Content']

    if (item_fat_content== 'Low Fat'):
        item_fat_content = 0,0
    elif (item_fat_content== 'Regular'):
        item_fat_content = 0,1
    else:
        item_fat_content = 1,0

    Item_Fat_Content_1,Item_Fat_Content_2 = item_fat_content

    Item_Visibility = float(request.form['Range 0.6-1.8'])

    Item_MRP = float(request.form['Item MRP'])
    
    Outlet_Identifier = request.form['Outlet ID']
    Outlet_ID = Outlet_Identifier
    if (Outlet_Identifier== 'OUT010'):
        Outlet_Identifier = 0,0,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT013'):
        Outlet_Identifier = 1,0,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT017'):
        Outlet_Identifier = 0,1,0,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT018'):
        Outlet_Identifier = 0,0,1,0,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT019'):
        Outlet_Identifier = 0,0,0,1,0,0,0,0,0
    elif (Outlet_Identifier== 'OUT027'):
        Outlet_Identifier = 0,0,0,0,1,0,0,0,0
    elif (Outlet_Identifier== 'OUT035'):
        Outlet_Identifier = 0,0,0,0,0,1,0,0,0
    elif (Outlet_Identifier== 'OUT045'):
        Outlet_Identifier = 0,0,0,0,0,0,1,0,0                        
    elif (Outlet_Identifier== 'OUT046'):
        Outlet_Identifier = 0,0,0,0,0,0,0,1,0       
    else:
        Outlet_Identifier = 0,0,0,0,0,0,0,0,1

    Outlet_1, Outlet_2,Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8,Outlet_9 = Outlet_Identifier



    Outlet_Year = int(2013 - int(request.form['Year']))

    Outlet_Size  = request.form['Size']
    if (Outlet_Size == 'Medium'):
        Outlet_Size = 1,0
    elif (Outlet_Size == 'Small'):
        Outlet_Size = 0,1
    else:
        Outlet_Size = 0,0

    Outlet_Size_1, Outlet_Size_2 = Outlet_Size

    Outlet_Location_Type = request.form['Location Type']
    if (Outlet_Location_Type == 'Tier 2'):
        Outlet_Location_Type = 1,0
    elif (Outlet_Location_Type == 'Tier 3'):
        Outlet_Location_Type = 0,1
    else:
        Outlet_Location_Type = 0,0

    Outlet_Location_Type_1,Outlet_Location_Type_2 = Outlet_Location_Type    

    Outlet_Type = request.form['Outlet Type']
    if (Outlet_Type == 'Supermarket Type1'):
        Outlet_Type = 1,0,0
    elif (Outlet_Type == 'Grocery Store'):
        Outlet_Type = 0,0,0
    elif (Outlet_Type == 'Supermarket Type3'):
        Outlet_Type = 0,0,1
    else:
        Outlet_Type = 0,1,0

    Outlet_Type_1, Outlet_Type_2, Outlet_Type_3 = Outlet_Type   

    Item_Type_Combined = request.form['Item Type']
    
    if(Item_Type_Combined == "Drinks"):
        Item_Type_Combined = 0,0
    elif (Item_Type_Combined == "Food"):
        Item_Type_Combined = 1,0
    else:
        Item_Type_Combined = 0,1    

    Item_Type_Combined_1, Item_Type_Combined_2 = Item_Type_Combined

    data = [Item_Weight, Item_Visibility, Item_MRP, Outlet_Year,Item_Fat_Content_1, Item_Fat_Content_2, Outlet_Location_Type_1,Outlet_Location_Type_2, Outlet_Size_1, Outlet_Size_2,Outlet_Type_1, Outlet_Type_2, Outlet_Type_3,Item_Type_Combined_1, Item_Type_Combined_2, Outlet_1, Outlet_2,Outlet_3, Outlet_4, Outlet_5, Outlet_6, Outlet_7, Outlet_8,Outlet_9]
    features_value = [np.array(data)]

    features_name = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years',
    'Item_Fat_Content_1', 'Item_Fat_Content_2', 'Outlet_Location_Type_1',
    'Outlet_Location_Type_2', 'Outlet_Size_1', 'Outlet_Size_2',
    'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3',
    'Item_Type_Combined_1', 'Item_Type_Combined_2', 'Outlet_1', 'Outlet_2',
    'Outlet_3', 'Outlet_4', 'Outlet_5', 'Outlet_6', 'Outlet_7', 'Outlet_8',
    'Outlet_9']

    df = pd.DataFrame(features_value, columns=features_name)

    myprd = model.predict(df)
    output=round(myprd[0],2)

    # if output < 0:
    #     return render_template('index.html',prediction_texts="Sorry you cannot sell")
    # else:
    #     return render_template('index.html',prediction_text="Item_Outlet_Sales at {}".format(output))

    return render_template('result.html',prediction = output,Item_Identifier = Item_Identifier, Outlet_Identifier = Outlet_ID)
    #return render_template('result.html', prediction_text='The Sales production of {} by {} is  {}  price'.format(Item_Identifier,Outlet_ID,output))


if __name__ == '__main__':
	app.run(debug=True)
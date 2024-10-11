from flask import Flask, redirect, request, session, url_for, render_template,flash
from flask_session import Session
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import h5py
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from markupsafe import Markup
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import urllib.parse
from urllib.parse import urljoin
import plotly.express as px
from plotly import graph_objs as go
from flask_mysqldb import MySQL 
import string , random


seq_length = 90
data_set = pd.read_csv("remaining_stock_data.csv")
actual_data_last_month=pd.read_csv('last_month_data.csv',index_col=0)
actual_data_last_month.index=pd.to_datetime(actual_data_last_month.index)
scaler = MinMaxScaler()
df=data_set.copy()
df['tiker'] = df['tiker'].astype(str)
df['date']=pd.to_datetime(df['date'])
df.sort_index(ascending=True,inplace=True)
#set datetime as dataset index
df.set_index('date', inplace=True)
df.drop(['open','low','volume','high'],axis=1,inplace=True)
df.index = pd.to_datetime(df.index)
lst_stock = []



app = Flask(__name__)
app.config['SECRET_KEY'] = '1234567890'

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'stocks market'

mysql = MySQL(app)


def scale_data(stock_df):
    data = stock_df.values
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    tr_size = int(np.ceil(len(scaled_data) * 0.90))
# Split the data into train and test sets
    splite_data = scaled_data[0:tr_size,0:1]
    test_data = scaled_data[len(splite_data):,0:1]
    # here we well return test data for take last 90 days to make future forcastingS
    return test_data
    
loaded_models = {}
# Load model
def load_saved_model():
    with h5py.File('Allstockmodels.keras', 'r') as hf:
        for symbol in hf.keys():
            lst_stock.append(symbol)
            model_filename = hf[symbol]['model'][()]
            print(model_filename)
            if isinstance(model_filename, bytes):
                model_filename = model_filename.decode('utf-8')

            model_path = os.path.join('stock_already_train', os.path.basename(model_filename))
            model = keras.models.load_model(model_path)
            loaded_models[symbol] = model
    return loaded_models

# Filter data for the current stock symbol
def filter_data_for_test(symbol):
    df_stock_s = df[df['tiker'] == symbol]
    df_stock = pd.DataFrame(df_stock_s['close'])
    data_scaled = scale_data(df_stock)
    return data_scaled, df_stock

def shift_and_insert(Xin, new_input):
    for i in range(seq_length - 1):
        Xin[:, i, :] = Xin[:, i + 1, :]
    Xin[:, seq_length - 1, :] = new_input
    return Xin


def make_forecasting(input_data, symbol, df_stock_s,future_days_forcasting=30):
    forecasted_values = []
    timestamps = []
    for i in range(0, future_days_forcasting):
        next_prediction = loaded_models[symbol].predict(input_data, batch_size=3,verbose=0)
        forecasted_values.append(next_prediction[0, 0])
        input_data = shift_and_insert(input_data, next_prediction[0, 0])
        timestamps.append(pd.to_datetime(df_stock_s.index[-1]) + timedelta(days=i))
    return timestamps, forecasted_values
def dataframe_result_forecasting(symbol,future_steps):
    test_data, df_stock = filter_data_for_test(symbol)
    X_input = test_data[-seq_length:].reshape(1, seq_length, 1)
    timestamps, forecasted_values = make_forecasting(X_input, symbol, df_stock,future_steps)
    forecasted_output = np.asarray(forecasted_values)
    forecasted_output = forecasted_output.reshape(-1, 1)
    forecasted_output = scaler.inverse_transform(forecasted_output)
    forecasted_output = pd.DataFrame(forecasted_output)
    date = pd.DataFrame(timestamps)
    df_result_forecasted = pd.concat([date, forecasted_output], axis=1)
    df_result_forecasted.columns = ["date", "Forecasted"]
    df_result_forecasted["tiker"] = symbol
    df_rsult=evaluate_and_compare_forecast(df_result_forecasted,symbol)
    result_json = df_rsult.to_json(orient='records')
    plot_data=create_forecast_plot(df_rsult,symbol)
    return plot_data, result_json



def create_forecast_plot(df_rsult, symbol):
    # إنشاء الرسم البياني مع السلسلة المتوقعة
    result_json = df_rsult.to_json(orient='records')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_rsult.index,
        y=df_rsult['Forecasted'],
        name="Forecasted Price",
        
        line=dict(color='#FE0002'),
        #stackgroup="price"
        legendgroup='price'
        
    )) 
    
    fig.add_trace(go.Scatter(
        x=df_rsult.index,
        y=df_rsult['Actual'],
        name="Actual Price",
        
        line=dict(color='orange'),
        #stackgroup="price"
        legendgroup='price'
    ))
   
    fig.update_layout(
        title_text=f"Forecast vs Actual Price for {symbol}",
        hovermode='x unified',
        title_font=dict(color='gray'),
        plot_bgcolor='#B7D0E1',
        paper_bgcolor='#CADEED'
    
        
    )
    
    plot_string = io.StringIO()
    fig.write_html(plot_string)
    plot_data = Markup(plot_string.getvalue())
    
    return plot_data


def evaluate_and_compare_forecast(df_result_forecasted, symbol):
  
    actual_data_stock = actual_data_last_month[actual_data_last_month['tiker']==symbol][['close','tiker']]
    actual_data_stock.index = pd.to_datetime(actual_data_stock.index)
    df_result_forecasted.set_index('date', inplace=True)
    df_result_forecasted.index = pd.to_datetime(df_result_forecasted.index)  
    df_resampled = actual_data_stock.resample('D').interpolate()
    df_result = pd.merge(df_result_forecasted, df_resampled, left_index=True, right_index=True, how='left')
    df_result.rename(columns={'close': 'Actual'}, inplace=True)
    df_result['Actual'] = df_result['Actual'].fillna(method='bfill')
    # Calculate evaluation metrics: MAE, MSE, and MAPE
    mae = mean_absolute_error(df_result['Actual'], df_result['Forecasted'])
    mse = mean_squared_error(df_result['Actual'], df_result['Forecasted'])
    mape = np.mean(np.abs((df_result['Actual'] - df_result['Forecasted']) / df_result['Actual'])) * 100
    print('ths mae',mae,'mse',mse)
    df_result.drop('tiker_y',axis=1,inplace=True)
# Correct the code
    df_result.rename(columns={'tiker_x': 'tiker'}, inplace=True)

# Set 'tiker' as the first column
    cols = list(df_result.columns)
    cols = [cols[-1]] + cols[:-1]
    df_result = df_result[cols]

    # tiker_y
    
    return df_result

def fetch_data(userID):

    cur = mysql.connection.cursor()
    cur.execute(f"""select prediction.predictionID, stock_symbol, stock_name ,interval_by_days, currency, created_date 
                from prediction
                INNER JOIN stock ON prediction.stockID = stock.stockID
                INNER JOIN saved_predictions ON prediction.predictionID = saved_predictions.predictionID
                where saved_predictions.userID = '{userID}'; """)
    rows = cur.fetchall()
    cur.close()

    data_list = []
    for row in rows:
        data_dict = {}
        data_dict['ID'] = row[0]  
        data_dict['Stock Sympol'] = row[1]  
        data_dict['Stock Name'] = row[2] 
        data_dict['interval'] = row[3] 
        data_dict['currency'] = row[4] 
        data_dict['Date'] = row[5] 

        data_list.append(data_dict)
    row_id=0
    table_html = '<table border="1">'
    table_html += '<tr><th>#</th><th>Stock Sympol</th><th>Stock Name</th><th>Interval</th><th>Currency</th><th>Saved Date </th><th></th><th></th></tr>'
    for row in data_list:
        row_id +=1
        table_html += '<tr>'
        table_html += f'<td>{row_id}</td>'
        table_html += f'<td>{row["Stock Sympol"]}</td>'
        table_html += f'<td>{row["Stock Name"]}</td>'
        table_html += f'<td>{row["interval"]} Days</td>'
        table_html += f'<td>{row["currency"]}</td>'
        table_html += f'<td>{row["Date"]}</td>'
        table_html += f'<td><a href="#" style="background:#194569" class="show" data-value="{row["ID"]}">Show</a></td>'
        table_html += f'<td><a href="#" style="background:red" class="del" data-value="{row["ID"]}" >Delete</a></td>'
        table_html += '</tr>'
        
    table_html += '</table>'

    return table_html

def getuser():
    email = session['email']
    cur2 = mysql.connection.cursor()
    cur2.execute(f"select userID from user where email ='{email}'")
    userID = cur2.fetchone()[0]
    cur2.close()
    return userID



with app.app_context():
    load_saved_model()


sym=""

@app.route('/', methods=['GET', 'POST'])
def main():
    if 'email' in session:

        cur2 = mysql.connection.cursor()
        cur2.execute(f"select stock_name , stock_symbol from stock ")
        stock = cur2.fetchall()
        cur2.close()
        return render_template('index.html',stock=stock ,email=session['email'])
    else:
        return render_template('login.html')
       

@app.route('/about', methods=['POST', 'GET'])
def about_us():
    if 'email' in session:
        return render_template('about.html',email=session['email'])
    return render_template('about.html')

@app.route('/contact', methods=['POST', 'GET'])
def contact_us():
    if 'email' in session:
        if request.method == 'POST':
            subject = request.form['subject']
            message = request.form['message']
            user =getuser()

            cur = mysql.connection.cursor()
            cur.execute(f"insert into contact (userID , subject , message) values ('{user}','{subject}','{message}')")
            mysql.connection.commit()
            cur.close()

        return render_template('contact.html',email=session['email'])
    return redirect(url_for('login'))  

@app.route('/login', methods=['GET', 'POST'])
def login():
   if request.method == 'POST':
        email =request.form['email']
        pwd = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute(f"select email , password from user where email ='{email}'")
        user = cur.fetchone()
        cur.close()
        if user and pwd == user[1]:
            session['email'] = user[0]
            return redirect(url_for('main'))
        else:
            flash("invalid email address or password")
        
   return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
       
        fname = request.form['firstname']
        lname = request.form['lastname'] 
        username = request.form['username']
        email = request.form['email']
        pwd = request.form['password']
        flag = True
        flag2 =True
        try:
            cur = mysql.connection.cursor()
            cur.execute(f"select email  from user where email ='{email}'")
            a=cur.fetchone()[0]
            cur.close()
            flash('The Email is alredy Exist !')
            flag=False
        except TypeError:
            flag =True
        
        try:
            cur3 = mysql.connection.cursor()
            cur3.execute(f"select username from user where username ='{username}'")
            b=cur3.fetchone()[0]
            cur3.close()            
            flash('The UserName is alredy Exist !')
            flag2=False
        except TypeError:
            flag2=True

        if flag and flag2:
            cur2 = mysql.connection.cursor()
            cur2.execute(f"""insert into user (first_name , last_name , username ,email , password) 
                         values ('{fname}','{lname}','{username}','{email}','{pwd}')""")
            mysql.connection.commit()
            cur2.close()
            return redirect(url_for('login'))    
     
    return render_template('register.html')


@app.route('/forgotPassword', methods = ['GET', 'POST'])
def forgotPassword():
    if request.method == 'POST':
    
       characters = string.ascii_letters + string.digits + string.punctuation
    # Generate the random string
       random_string = ''.join(random.choice(characters) for _ in range(10))

       email= request.form['loginEmail']
       pwd = random_string

       cur = mysql.connection.cursor()
       cur.execute(f"update user set password='{pwd}' where email ='{email}'")
       mysql.connection.commit()
       cur.close()
    return render_template('forgotPassword.html')

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('email', None)
   return redirect(url_for('main'))


@app.route('/forecast', methods=['GET', 'POST'])
def forecasting():
    if 'email' in session:
        if request.method == 'POST':
            symbol = request.form['symbol']
            future_steps=request.form['future_days']
            future_steps = int(future_steps)
            plot_data, result_json = dataframe_result_forecasting(symbol,future_steps)
            final_df = pd.read_json(result_json)
            global sym
            sym = symbol
            return render_template('forecast.html',
                         plot=plot_data, data=final_df.to_html(),email=session['email'])
        return render_template('forecst.html',email=session['email'])
    else:
        return redirect(url_for('login'))
  
  
@app.route('/saved', methods=['POST', 'GET'])
def saved():
    if 'email' in session:

       global sym

       cur1 = mysql.connection.cursor()
       cur1.execute(f"""select predictionID 
                    from prediction 
                    inner join stock on prediction.stockID = stock.stockID
                    where stock_symbol ='{sym}'""")
       predID = cur1.fetchone()[0]
       cur1.close()

       userID = getuser()

       cur3 = mysql.connection.cursor()
       cur3.execute(f"insert into saved_predictions (predictionID, userID) values ('{predID}','{userID}')")
       mysql.connection.commit()
       cur3.close()
       return redirect(url_for('gotosaved'))

    return redirect(url_for('login'))



@app.route('/gotosaved', methods=['GET', 'POST'])
def gotosaved():
    if 'email' in session:
        userID = getuser()
        data_table=fetch_data(userID)
        return render_template('saved.html',table=data_table ,email=session['email'])
    
@app.route('/showdatasaved', methods=['GET', 'POST'])
def showsaved():
    if 'email' in session:

        preID = request.form.get('show')

        cur = mysql.connection.cursor()
        cur.execute(f""" select stock_symbol , interval_by_days 
                    from prediction
                    INNER JOIN stock ON prediction.stockID = stock.stockID
                    where predictionID='{preID}'; """)
        result = cur.fetchone()
        cur.close()

        symbol = result[0]
        future_steps = result[1]
        future_steps = int(future_steps)
        plot_data, result_json = dataframe_result_forecasting(symbol,future_steps)
        final_df = pd.read_json(result_json)

        return render_template('showdatasaved.html',plot=plot_data, data=final_df.to_html(),email=session['email'])

    return redirect(url_for('login'))


@app.route('/deletesaved', methods=['GET', 'POST'])
def deletesaved():
    if 'email' in session:
        preID = request.form.get('del')

        cur = mysql.connection.cursor()
        cur.execute(f"DELETE FROM saved_predictions WHERE predictionID = {preID};")
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('gotosaved'))
    return redirect(url_for('login'))
    
if __name__ == '__main__':
    app.run(debug=True)
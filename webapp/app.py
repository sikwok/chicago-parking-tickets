from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import reset_output
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import Label
import datetime

import simplejson as json
import requests
import os

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

import ast

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def my_form_post():
    if request.method == 'GET':
        return render_template('my-form.html')

    elif request.method == 'POST':
        cutoff1 = float(request.form['cutoff1'])
        cutoff2 = float(request.form['cutoff2'])
        violdic = ast.literal_eval(request.form['violationcode'])
        violcode = violdic['code']
        violdesc = violdic['desc']
        if cutoff2 < cutoff1 or cutoff1 < 0 or cutoff2 < 0:
            return render_template('badrequest.html', co1 = cutoff1, co2 = cutoff2)
        
        df = pd.read_csv("chicago_averages.csv")
        df_temp = df.drop_duplicates(subset = {'zipcode','violation_code'}).loc[df['violation_code'] == violcode][['avg_initial_fine_zipcode_viol','log_income','zipcode_viol_count']]
        df_temp = df_temp.loc[df_temp['zipcode_viol_count']>=cutoff1]
        df_temp = df_temp.loc[df_temp['zipcode_viol_count']<=cutoff2]
        if len(df_temp)<2:
            return render_template('notenoughobs.html', co1 = cutoff1, co2 = cutoff2)
        y = df_temp['avg_initial_fine_zipcode_viol'].values.reshape(-1,1)
        X = df_temp[['log_income']].values.reshape(-1,1)
        
        results = []
        grid = np.linspace(0.5,20,400)
        for alpha in grid:
            scores = cross_val_score(Ridge(alpha), X, y, cv=3)
            results.append(scores.mean()) 
        alpha_optimal_index = np.argmax(results)
        alpha_optimal2 = grid[alpha_optimal_index]
        alpha_4 = alpha_optimal2
        pipe_4 = Ridge(alpha = alpha_4)
        pipe_4.fit(X,y)
        grid_test = np.linspace(9.5,12,1000).reshape(-1,1)
        grid_result = pipe_4.predict(grid_test)
        
        p = figure(plot_width=800, plot_height=250, title = "Initial Fine for %s vs. Zip Code Income" %(violdesc))
        p.line(grid_test.flatten(), grid_result.flatten(), color='navy', alpha=0.5)
        p.scatter(X.flatten(),y.flatten())
        p.xaxis.axis_label = "Log of Zip Code Median Household Income (2006-2010)"
        p.yaxis.axis_label = "Initial Fine Amount ($)"
        script, div = components(p)
        return render_template('graph.html', script=script, div=div)
        
    else:
        return render_template('helloworld.html')
    
if __name__ == '__main__':
    app.run(port=33507)

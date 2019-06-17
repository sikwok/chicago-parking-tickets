from __future__ import division
from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import reset_output, show
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import Label, ColumnDataSource, LayoutDOM
from bokeh.core.properties import Instance, String
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


JS_CODE = """
# This file contains the JavaScript (CoffeeScript) implementation
# for a Bokeh custom extension. The "surface3d.py" contains the
# python counterpart.
#
# This custom model wraps one part of the third-party vis.js library:
#
#     http://visjs.org/index.html
#
# Making it easy to hook up python data analytics tools (NumPy, SciPy,
# Pandas, etc.) to web presentations using the Bokeh server.

# These "require" lines are similar to python "import" statements
import * as p from "core/properties"
import {LayoutDOM, LayoutDOMView} from "models/layouts/layout_dom"

# This defines some default options for the Graph3d feature of vis.js
# See: http://visjs.org/graph3d_examples.html for more details.
OPTIONS =
  width:  '700px'
  height: '700px'
  style: 'dot-color'
  showPerspective: true
  showGrid: true
  keepAspectRatio: true
  verticalRatio: 1.0
  showLegend: false
  cameraPosition:
    horizontal: -0.35
    vertical: 0.22
    distance: 1.8
  dotSizeRatio: 0.01
  tooltip: (point) -> return 'value: <b>' + point.z + '</b><br>' + 'x: <b>' + point.x + '</b><br>' + 'y: <b>' + point.y




# To create custom model extensions that will render on to the HTML canvas
# or into the DOM, we must create a View subclass for the model. Currently
# Bokeh models and views are based on BackBone. More information about
# using Backbone can be found here:
#
#     http://backbonejs.org/
#
# In this case we will subclass from the existing BokehJS ``LayoutDOMView``,
# corresponding to our
export class Surface3dView extends LayoutDOMView

  initialize: (options) ->
    super(options)

#    url = "http://visjs.org/dist/vis.js"
    url = "static/vis.js"
    script = document.createElement('script')
    script.src = url
    script.async = false
    script.onreadystatechange = script.onload = () => @_init()
    document.querySelector("head").appendChild(script)

  _init: () ->
    # Create a new Graph3s using the vis.js API. This assumes the vis.js has
    # already been loaded (e.g. in a custom app template). In the future Bokeh
    # models will be able to specify and load external scripts automatically.
    #
    # Backbone Views create <div> elements by default, accessible as @el. Many
    # Bokeh views ignore this default <div>, and instead do things like draw
    # to the HTML canvas. In this case though, we use the <div> to attach a
    # Graph3d to the DOM.
    @_graph = new vis.Graph3d(@el, @get_data(), OPTIONS)

    # Set Backbone listener so that when the Bokeh data source has a change
    # event, we can process the new data
    @connect(@model.data_source.change, () =>
        @_graph.setData(@get_data())
    )

  # This is the callback executed when the Bokeh data has an change. Its basic
  # function is to adapt the Bokeh data source to the vis.js DataSet format.
  get_data: () ->
    data = new vis.DataSet()
    source = @model.data_source
    for i in [0...source.get_length()]
      data.add({
        x:     source.get_column(@model.x)[i]
        y:     source.get_column(@model.y)[i]
        z:     source.get_column(@model.z)[i]
        style: source.get_column(@model.color)[i]
      })
    return data

# We must also create a corresponding JavaScript Backbone model sublcass to
# correspond to the python Bokeh model subclass. In this case, since we want
# an element that can position itself in the DOM according to a Bokeh layout,
# we subclass from ``LayoutDOM``
export class Surface3d extends LayoutDOM

  # This is usually boilerplate. In some cases there may not be a view.
  default_view: Surface3dView

  # The ``type`` class attribute should generally match exactly the name
  # of the corresponding Python class.
  type: "Surface3d"

  # The @define block adds corresponding "properties" to the JS model. These
  # should basically line up 1-1 with the Python model class. Most property
  # types have counterparts, e.g. ``bokeh.core.properties.String`` will be
  # ``p.String`` in the JS implementatin. Where the JS type system is not yet
  # as rich, you can use ``p.Any`` as a "wildcard" property type.
  @define {
    x:           [ p.String           ]
    y:           [ p.String           ]
    z:           [ p.String           ]
    color:       [ p.String           ]
    data_source: [ p.Instance         ]
  }
"""

# This custom extension model will have a DOM view that should layout-able in
# Bokeh layouts, so use ``LayoutDOM`` as the base class. If you wanted to create
# a custom tool, you could inherit from ``Tool``, or from ``Glyph`` if you
# wanted to create a custom glyph, etc.
class Surface3d(LayoutDOM):

    # The special class attribute ``__implementation__`` should contain a string
    # of JavaScript (or CoffeeScript) code that implements the JavaScript side
    # of the custom extension model.
    __implementation__ = JS_CODE

    # Below are all the "properties" for this model. Bokeh properties are
    # class attributes that define the fields (and their types) that can be
    # communicated automatically between Python and the browser. Properties
    # also support type validation. More information about properties in
    # can be found here:
    #
    #    https://bokeh.pydata.org/en/latest/docs/reference/core.html#bokeh-core-properties

    # This is a Bokeh ColumnDataSource that can be updated in the Bokeh
    # server by Python code
    data_source = Instance(ColumnDataSource)

    # The vis.js library that we are wrapping expects data for x, y, z, and
    # color. The data will actually be stored in the ColumnDataSource, but
    # these properties let us specify the *name* of the column that should
    # be used for each field.
    x = String
    y = String
    z = String
    color = String



@app.route('/', methods=['POST','GET'])
def my_form_post():
#    try:
    if request.method == 'GET':
        return render_template('my-form.html')

    elif request.method == 'POST':
        cutoff1 = float(request.form['cutoff1'])
        violdic = ast.literal_eval(request.form['violationcode'])
        violcode = violdic['code']
        violdesc = violdic['desc']
        groupmetricdic = ast.literal_eval(request.form['groupmetric'])
        groupcode = groupmetricdic['code']
        groupdesc = groupmetricdic['desc']      
        df = pd.read_csv("chicago_averages.csv")
        
        
        df_temp = df.drop_duplicates(subset = {'zipcode','violation_code'}).loc[df['violation_code'] == violcode][['avg_fine_viol_zipcode','log_income','proportion_black','zipcode_viol_count','year']]
        df_temp = df_temp.loc[df_temp['zipcode_viol_count']>=cutoff1]
        if len(df_temp)<2:
            return render_template('notenoughobs.html', co1 = cutoff1)
        y = df_temp['avg_fine_viol_zipcode'].values.reshape(-1,1)
        
        if groupcode!="both":
            X1 = df_temp[['log_income']].values.reshape(-1,1)
            X2 = df_temp[['proportion_black']].values.reshape(-1,1)
            X = df_temp[['log_income', 'proportion_black']].values
            
            
            X = df_temp[[groupcode]].values.reshape(-1,1)
            results = []
            grid = np.linspace(0.5,20,100)
            for alpha in grid:
                scores = cross_val_score(Ridge(alpha), X, y, cv=3)
                results.append(scores.mean()) 
            alpha_optimal_index = np.argmax(results)
            alpha_optimal2 = grid[alpha_optimal_index]
            alpha_4 = alpha_optimal2
            pipe_4 = Ridge(alpha = alpha_4)
            pipe_4.fit(X,y)
            theta = round(pipe_4.coef_.flatten()[0],4)
            r_squared = round(pipe_4.score(X,y),4)
            
            
            if groupcode == "proportion_black":
                grid_test = np.linspace(0.0,1.0,100).reshape(-1,1)
            elif groupcode == "log_income":
                grid_test = np.linspace(9.5,12,1000).reshape(-1,1)
            out = pipe_4.predict(grid_test)
            p = figure(plot_width=800, plot_height=250, title = "Initial Fine for %s vs. %s" %(violdesc, groupdesc))
            p.line(grid_test.flatten(), out.flatten(), color='navy', alpha=0.5)
            p.scatter(X.flatten(),y.flatten())
            if groupdesc == "proportion_black":    
                p.xaxis.axis_label = "Proportion of Black Population in Zip Code (2010 U.S. Census)"
            elif groupdesc == "log_income":
                p.xaxis.axis_label = "Log of Zip Code Median Household Income (2006-2010 American Community Survey)"
            p.yaxis.axis_label = "Initial Fine Amount ($)"
            script, div = components(p)
            return render_template('graph.html', script=script, div=div, theta = theta, r_squared = r_squared)
        elif groupcode=='both':
            X = df_temp[['log_income', 'proportion_black']].values
            results = []
            grid = np.linspace(0.5,20,100)
            for alpha in grid:
                scores = cross_val_score(Ridge(alpha), X, y, cv=3)
                results.append(scores.mean()) 
            alpha_optimal_index = np.argmax(results)
            alpha_optimal2 = grid[alpha_optimal_index]
            alpha_4 = alpha_optimal2
            pipe_4 = Ridge(alpha = alpha_4)
            pipe_4.fit(X,y)
            theta_income = round(pipe_4.coef_.flatten()[0],4)
            theta_blackprop = round(pipe_4.coef_.flatten()[1],4)
            r_squared = round(pipe_4.score(X,y),4)
            X_data = X[:,0]
            Y_data = X[:,1]
            Z_data = y
            color = np.asarray([0 for x in range(len(X_data))])
            
            source = ColumnDataSource(data=dict(x=X_data, y=Y_data, z=Z_data.flatten(), color = color))        
            surface = Surface3d(x="x", y="y", z="z", color="color", data_source=source)
            script, div = components(surface)
            return render_template('graph2.html', script=script, div=div, violdesc = violdesc, theta_income = theta_income, theta_blackprop = theta_blackprop, r_squared = r_squared)
        
    
if __name__ == '__main__':
    app.run(port=33507)

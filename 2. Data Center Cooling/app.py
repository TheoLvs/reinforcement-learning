#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
TWITTER APP

Started on the 22/06/2017


https://plot.ly/dash/live-updates
https://plot.ly/dash/getting-started
https://plot.ly/dash/getting-started-part-2
https://plot.ly/dash/gallery/new-york-oil-and-gas/

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

# USUAL
import os


# DASH IMPORT
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import plotly.graph_objs as go

# EKIMETRICS IMPORT
import json
config = json.loads(open("config.json","r").read())
ekimetrics_path = config["ekimetrics_path"]

import sys
sys.path.append(ekimetrics_path)

from ekimetrics.api import twitter



#---------------------------------------------------------------------------------
# CREATE THE APP
app = dash.Dash("Twitter Listener")

# # Making the app available offline
offline = False
app.css.config.serve_locally = offline
app.scripts.config.serve_locally = offline


style = {
    'font-weight': 'bolder',
    'font-family': 'Product Sans',
    }






#---------------------------------------------------------------------------------
# LAYOUT
app.layout = html.Div(children=[





    # HEADER FIRST CONTAINER
    html.Div([
        html.Img(src="https://abs.twimg.com/icons/apple-touch-icon-192x192.png",style={'height': '45px','float': 'left'}),
        html.H1("Twitter Listener",style = {'color': "rgba(117, 117, 117, 0.95)",**style}),
        dcc.Input(id='keywords-input', value='macron', type="text"),
        html.P(id='keywords',style = {"display":"inline","font-size":"0.8em"}),
        html.Br([]),
        html.Br([]),
        html.Button("Streaming",id = "streaming-button",style = style),
        html.Button("Reloading",id = "reloading-button",style = style),
        html.H4("0 Tweets retrieved",id = "count-tweets"),
        html.Img(id = "loading-image",src="https://img.artlebedev.ru/everything/ib-translations/site/process/ibt-process-42.gif",style={'height': '50px',"display":"none"}),
        dcc.Interval(id='interval-component',interval=1*1000*100), # in milliseconds)
    ],style=style, className="container"),


    # ANALYTICS CONTAINER
    html.Div([

        dcc.Graph(id='count-words',animate = False),

    ],style=style, className="container"),


])




#---------------------------------------------------------------------------------
# CALLBACKS


# Callback for input search
@app.callback(
    Output(component_id='keywords', component_property='children'),
    [Input(component_id='keywords-input', component_property='value')]
)
def update_output_div(input_value):
    return ' = Keywords : {}'.format(input_value.split(","))


# Launch the object streamer
streamer = twitter.Twitter_Streamer()



# Callback to stop the streaming
@app.callback(
    Output("loading-image","style"),
    events = [Event('streaming-button', 'click')])
def streaming():
    if not streamer.is_streaming:
        return {'height': '50px',"display":"inline"}
    else:
        return {'height': '50px',"display":"none"}



# Callback to stop the streaming
@app.callback(
    Output("interval-component","interval"),
    events = [Event('streaming-button', 'click')],
    state = [State("keywords-input","value")])
def refreshing(input_value):
    if not streamer.is_streaming:
        print("start streaming ...")
        streamer.streaming(input_value.split(","))
        return 1*1000*10
    else:
        streamer.disconnect()
        print("stopping streaming ...")
        return 1*1000*100





# Automatic callback to update the word count during streaming
@app.callback(
    Output('count-words', 'figure'),
    events=[Event('interval-component', 'interval'),Event('reloading-button', 'click')],
    state = [State("keywords-input","value")])
def update_count_words(input_value):
    file_path = "twitter_data/" + input_value.replace(",","_")+".txt"
    if os.path.exists(file_path):
        loader = twitter.Twitter_Loader(file_path = file_path)
        tweets = twitter.Tweets(json_data = loader.data,verbose = 0)
        tweets.filter(language = ["english","french"])
        df = tweets.count_words().head(10)
        x = list(df.index)
        y = list(df["count"])
    else:
        x = []
        y = []

    fig = [go.Bar(x = x,y = y)]
    return {"data":fig}




# Automatic callback to update the number of tweets streamed
@app.callback(
    Output('count-tweets', 'children'),
    events=[Event('interval-component', 'interval'),Event('reloading-button', 'click')],
    state = [State("keywords-input","value")])
def update_count_tweets(input_value):
    if not streamer.is_streaming: print("REFRESHING ...")
    file_path = "twitter_data/" + input_value.replace(",","_")+".txt"
    if os.path.exists(file_path):
        loader = twitter.Twitter_Loader(file_path = file_path)
        count = len(loader.data)
    else:
        count = 0

    return "{} tweets retrieved".format(count)







#---------------------------------------------------------------------------------
# ADD EXTERNAL CSS

external_css = ["https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css"]

for css in external_css:
    app.css.append_css({"external_url": css})







#---------------------------------------------------------------------------------
# RUN SERVER
if __name__ == '__main__':
    app.run_server(debug=True)

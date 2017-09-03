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


import sys
sys.path.append("C:/git/reinforcement-learning/")




#--------------------------------------------------------------------------------
from rl.env.data_center_cooling import DataCenterCooling
from rl.agents.q_agent import QAgent
from rl.agents.dqn_agent import DQNAgent


env = DataCenterCooling()



#---------------------------------------------------------------------------------
# CREATE THE APP
app = dash.Dash("Data Cooling Center")

# # Making the app available offline
offline = False
app.css.config.serve_locally = offline
app.scripts.config.serve_locally = offline


style = {
    'font-weight': 'bolder',
    'font-family': 'Product Sans',
    }

container_style = {
    "margin":"20px",
}



AGENTS = [{"label":x,"value":x} for x in ["Q Agent","Deep-Q-Network Agent","Policy Gradient Agent"]]


#---------------------------------------------------------------------------------
# LAYOUT
app.layout = html.Div(children=[





    # HEADER FIRST CONTAINER
    html.Div([
        html.H2("Data Center Cooling",style = {'color': "rgba(117, 117, 117, 0.95)",**style}),
        dcc.Dropdown(id = "input-agent",options = AGENTS,value = "Q Agent",multi = False),
        html.Br(),
        dcc.Slider(min=100,max=2500,step=50,value=500,id = "n-episodes"),
        html.Br(),
        dcc.Slider(min=100,max=2500,step=50,value=500,id = "gamma"),
        html.Br(),
        html.Button("Train",id = "training",style = style),
    ],style={**style,**container_style,'width': '20%',"height":"800px", 'float' : 'left', 'display': 'inline'}, className="container"),




    # ANALYTICS CONTAINER
    html.Div([

        dcc.Graph(id='states',animate = False,figure = env.render_states_plotly()),
        dcc.Graph(id='rewards',animate = False,figure = env.render_rewards_plotly()),


    ],style={**style,**container_style,'width': '55%',"height":"800px", 'float' : 'right', 'display': 'inline'}, className="container"),


])




#---------------------------------------------------------------------------------
# CALLBACKS








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

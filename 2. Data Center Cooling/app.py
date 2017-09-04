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
import numpy as np
from tqdm import tqdm
from copy import deepcopy

# DASH IMPORT
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import plotly.graph_objs as go


import sys
sys.path.append("C:/git/reinforcement-learning/")



#--------------------------------------------------------------------------------
from rl.envs.data_center_cooling import DataCenterCooling
from rl.agents.q_agent import QAgent
from rl.agents.dqn_agent import DQNAgent
from rl import utils




def run_episode(env,agent,max_step = 100,verbose = 1):

    s = env.reset()
    
    episode_reward = 0
    
    i = 0
    while i < max_step:
        
        # Choose an action
        a = agent.act(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agent.train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward




def run_n_episodes(env,n_episodes = 2000,lr = 0.8,gamma = 0.95):

    environment = deepcopy(env)
    
    # Initialize the agent
    states_size = len(env.observation_space)
    actions_size = len(env.action_space)

    agent = QAgent(states_size,actions_size,lr = lr,gamma = gamma)
    
    # Store the rewards
    rewards = []
    
    # Experience replay
    for i in tqdm(range(n_episodes)):
        
        # Run the episode
        environment,agent,episode_reward = run_episode(environment,agent,verbose = 0)
        rewards.append(episode_reward)
        
    
    # Plot rewards
    # utils.plot_average_running_rewards(rewards)
        
    return environment,agent,rewards
        

class Clicks(object):
    def __init__(self):
        self.count = 0

reset_clicks = Clicks()
train_clicks = Clicks()
env = DataCenterCooling()
np.random.seed()

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



AGENTS = [{"label":x,"value":x} for x in ["Q Agent","SARSA Agent","Deep-Q-Network Agent","Policy Gradient Agent"]]

#---------------------------------------------------------------------------------
# LAYOUT
app.layout = html.Div(children=[





    # HEADER FIRST CONTAINER
    html.Div([
        html.H2("Data Center Cooling",style = {'color': "rgba(117, 117, 117, 0.95)",**style}),

        html.Div([
            html.H4("Environment",style = {'color': "rgba(117, 117, 117, 0.95)",**style}),
            html.P("Cooling levels",id = "cooling"),
            dcc.Slider(min=10,max=100,step=10,value=10,id = "levels-cooling"),
            html.P("Cost factor",id = "cost-factor"),
            dcc.Slider(min=0.0,max=5,step=0.1,value=1.0,id = "levels-cost-factor"),
            html.P("Risk factor",id = "risk-factor"),
            dcc.Slider(min=0.0,max=5,step=0.1,value=1.6,id = "levels-risk-factor"),    
            html.Br(),
            html.Button("Reset",id = "reset-env",style = style,n_clicks = 0),
        ],style = {"height":"50%"}),


        html.Div([
            html.H4("Agent",style = {'color': "rgba(117, 117, 117, 0.95)",**style}),
            dcc.Dropdown(id = "input-agent",options = AGENTS,value = "Q Agent",multi = False),
            html.P("N episodes",id = "input-episodes"),
            dcc.Slider(min=200,max=5000,step=200,value=1000,id = "n-episodes"),
            html.P("Learning rate",id = "input-lr"),
            dcc.Slider(min=0.001,max=1.0,step=0.005,value=0.1,id = "lr"),
            html.Br(),
            html.Button("Train",id = "training",style = style,n_clicks = 0),
        ],style = {"height":"50%"}),



    ],style={**style,**container_style,'width': '20%',"height":"800px", 'float' : 'left', 'display': 'inline'}, className="container"),




    # ANALYTICS CONTAINER
    html.Div([

        dcc.Graph(id='render',animate = False,figure = env.render(with_plotly = True),style = {"height":"100%"}),


    ],style={**style,**container_style,'width': '55%',"height":"800px", 'float' : 'right', 'display': 'inline'}, className="container"),


])




#---------------------------------------------------------------------------------
# CALLBACKS



# Callback to stop the streaming
@app.callback(
    Output("render","figure"),
    [Input('reset-env','n_clicks'),Input('training','n_clicks'),Input('levels-cost-factor','value'),Input('levels-risk-factor','value')],
    state = [State('levels-cooling','value'),State('lr','value'),State('n-episodes','value')]

    )
def render(click_reset,click_training,cost_factor,risk_factor,levels_cooling,lr,n_episodes):


    print("Reset ",click_reset," - ",reset_clicks.count)
    print("Train ",click_training," - ",train_clicks.count)


    if click_reset > reset_clicks.count:
        np.random.seed()
        reset_clicks.count = click_reset
        env.__init__(levels_cooling = levels_cooling,risk_factor = risk_factor,cost_factor = cost_factor)

    elif click_training > train_clicks.count:
        train_clicks.count = click_training
        print(env.risk_factor,env.cost_factor)
        env_temp,agent,rewards = run_n_episodes(env,n_episodes = n_episodes,lr = lr)
        env.cooling = env_temp.cooling
    else:
        env.risk_factor = risk_factor
        env.cost_factor = cost_factor



    return env.render(with_plotly = True)




@app.callback(
    Output("cooling","children"),
    [Input('levels-cooling','value')])
def update_cooling(value):
    env.levels_cooling = value
    env.define_cooling(value)
    return "Cooling levels : {}".format(value)



@app.callback(
    Output("risk-factor","children"),
    [Input('levels-risk-factor','value')])
def update_risk(value):
    return "Risk factor : {}".format(value)



@app.callback(
    Output("cost-factor","children"),
    [Input('levels-cost-factor','value')])
def update_cost(value):
    return "Cost factor : {}".format(value)

@app.callback(
    Output("input-episodes","children"),
    [Input('n-episodes','value')])
def update_episodes(value):
    return "N episodes : {}".format(value)

@app.callback(
    Output("input-lr","children"),
    [Input('lr','value')])
def update_lr(value):
    return "Learning rate : {}".format(value)






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
    np.random.seed()
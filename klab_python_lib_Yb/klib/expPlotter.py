from .imports import *
from .expPlotData import *

class ExpPlotter:

    def __init__(self, exp):
        if(exp):
            self.expP = ExpPlotData(exp)
        self.plotTTLChannels = []
        self.plotDACChannels = []
        self.plotDDSChannels = []

    def setExpPlotData(self, exp):
        self.expP = ExpPlotData(exp)

    def setTTLChannels(self, plotTTLChannels):
        self.plotTTLChannels = plotTTLChannels

    def setDACChannels(self, plotDACChannels):
        self.plotDACChannels = plotDACChannels

    def setDDSChannels(self, plotDDSChannels):
        self.plotDDSChannels = plotDDSChannels

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(('localhost', port)) == 0

def runExpViewer(expPlotter):

    if (expPlotter.expP == None):
        print('set the experiment with expPlotter.setExpPlotData(exp)')
        return

    numTTLC = len(expPlotter.plotTTLChannels)
    numDACC = len(expPlotter.plotDACChannels)
    numDDSC = len(expPlotter.plotDDSChannels)

    if (numTTLC == 0 or numDACC == 0 or numDDSC == 0):
        print('set at least one channel for TTL, DAC, and DDS')
        return

    app = JupyterDash(__name__)

    fig = subplots.make_subplots(rows=numTTLC, cols=1, shared_xaxes=True)
    for i, channel in enumerate(expPlotter.plotTTLChannels):
        fig.add_trace(go.Scatter(x=expPlotter.expP.TTLData['time'], 
        y=expPlotter.expP.TTLData[channel], fill='tozeroy',line=dict(width=0.5, color=cols[0]),
                        mode='lines'), row=i+1, col=1)
        fig.update_yaxes(range=[0, 1], row=i+1, col=1, tickvals=[0.5], ticktext=[channel])
    fig.update_layout(margin=dict(l=5, r=5, t=25, b=5), height=35*numTTLC)
    fig.update_layout(showlegend=False)

    fig2 = subplots.make_subplots(rows=numDACC, cols=1, shared_xaxes=True)
    for i, channel in enumerate(expPlotter.plotDACChannels):
        fig2.add_trace(go.Scatter(x=expPlotter.expP.DACData[channel]['time'], 
        y=expPlotter.expP.DACData[channel]['value'], 
                                fill='tozeroy',line=dict(width=0.5, color=cols[0]),
                        mode='lines'), row=i+1, col=1)
        fig2.update_yaxes(row=i+1, col=1, title_text=channel)
    fig2.update_layout(margin=dict(l=5, r=5, t=25, b=5), height=100*numDACC)
    fig2.update_layout(showlegend=False)

    fig3 = subplots.make_subplots(rows=numDDSC, cols=1, shared_xaxes=True)
    for i, channel in enumerate(expPlotter.plotDDSChannels):
        fig3.add_trace(go.Scatter(x=expPlotter.expP.DDSFreqData[channel]['time'], 
        y=expPlotter.expP.DDSFreqData[channel]['value'], 
                                fill='tozeroy', line=dict(width=1, color=cols[0]), mode='lines'), row=i+1, col=1)
        fig3.update_yaxes(row=i+1, col=1, title_text=channel)
    fig3.update_layout(margin=dict(l=5, r=5, t=25, b=5), height=100*numDDSC)
    fig3.update_layout(showlegend=False)

    fig4 = subplots.make_subplots(rows=numDDSC, cols=1, shared_xaxes=True)
    for i, channel in enumerate(expPlotter.plotDDSChannels):
        fig4.add_trace(go.Scatter(x=expPlotter.expP.DDSAmpData[channel]['time'], 
        y=expPlotter.expP.DDSAmpData[channel]['value'], 
                                fill='tozeroy',line=dict(width=0.5, color=cols[0]),
                        mode='lines'), row=i+1, col=1)
        fig4.update_yaxes(row=i+1, col=1, title_text=channel)
    fig4.update_layout(margin=dict(l=5, r=5, t=25, b=5), height=100*numDDSC)
    fig4.update_layout(showlegend=False)

    dcc.Graph(figure=fig)
    dcc.Graph(figure=fig2)
    dcc.Graph(figure=fig3)
    dcc.Graph(figure=fig4)

    app.layout = html.Div([
        html.Div([
        html.Div([html.Span(html.B('data address: ')),str(expPlotter.expP.data_addr)+'\data_'+str(expPlotter.expP.file_id)+'.h5  ', html.Br(),
                html.Span(html.B('date/time: ')), expPlotter.expP.experiment_date+' '+expPlotter.expP.experiment_time+'  ', html.Br(),
                html.Span(html.B('key names: [' )), " ,".join(expPlotter.expP.key_names)+'] ', html.Br(),
                html.Span(html.B('variations: ')), str(len(expPlotter.expP.key))+' ', html.Br(),
                ], style={'text-align': 'left', 'font-size': '12px'}),
        html.Br(),
        html.Div([
            html.Div('time range (ms)', style={'padding-right': '80px'}),
                html.Div(dcc.RangeSlider(
            expPlotter.expP.TTLData['time'].min(),
            expPlotter.expP.TTLData['time'].max(),
            step=None,
            id='time_slider',
            value=[expPlotter.expP.TTLData['time'].min(),expPlotter.expP.TTLData['time'].max()]
        ),style={'padding': '10px'})], style={'width': '90%', 'margin': '0 auto'})
        ], style={'position': 'fixed', 'top': '0', 'z-index': '100', 
                'background-color': '#fff',
                'padding': '20px 60px 20px 20px', 
                'width': '100%', 'border': '1px solid #ccc'}),
        html.Div([
        html.Div([
            # html.Div('test'),
            html.Div('TTL channels', style={'clear': 'both'})
        ], style={'padding-top': '200px'}),
        html.Div([
            dcc.Graph(figure=fig, id='ttl_plots'),
        ]),
        html.Br(),
        html.Div('DAC channels', style={'clear': 'both'}),
        html.Div([
            dcc.Graph(figure=fig2, id='dac_plots')
        ], style={'padding-left': '27px'}),
        html.Br(),
        html.Div('DDS channels (frequency)', style={'clear': 'both'}),
        html.Div([
            dcc.Graph(figure=fig3, id='dds_freq_plots')
        ], style={'padding-left': '27px'}),
        html.Br(),
        html.Div('DDS channels (amplitude)', style={'clear': 'both'}),
        html.Div([
            dcc.Graph(figure=fig4, id='dds_amp_plots')
        ], style={'padding-left': '27px'}),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    ], style={'margin': '20px 80px 0px 40px'})
        ], style={'padding': '0', 'margin': '0', 'font-family': 'sans-serif', 'text-align': 'center'})

    @app.callback(
        [Output('ttl_plots', 'figure'),
        Output('dac_plots', 'figure'),
        Output('dds_freq_plots', 'figure'),
        Output('dds_amp_plots', 'figure')],
        Input('time_slider', 'value'),
        State('ttl_plots', 'figure'),
        State('dac_plots', 'figure'),
        State('dds_freq_plots', 'figure'),
        State('dds_amp_plots', 'figure'))
    def update_plots(timeValue, figTTL, figDAC, figDDSFreq, figDDSAmp):
        f1 = go.Figure(figTTL)
        f1.update_xaxes({'range': timeValue, 'autorange': False})
        f2 = go.Figure(figDAC)
        f2.update_xaxes({'range': timeValue, 'autorange': False})
        f3 = go.Figure(figDDSFreq)
        f3.update_xaxes({'range': timeValue, 'autorange': False})
        f4 = go.Figure(figDDSAmp)
        f4.update_xaxes({'range': timeValue, 'autorange': False})
        return f1, f2, f3, f4

    # port = 8050


    # try:
    app.run_server(debug=True, mode='external')
    # except OSError:
    #     while (is_port_in_use(port) == True):
    #         port += 1
    #     app.run_server(debug=True, mode='external', port=port)
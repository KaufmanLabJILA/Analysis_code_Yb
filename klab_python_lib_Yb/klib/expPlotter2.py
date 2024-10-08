from .imports import *
####wouldnt work since pklotly is commented out when conda is upgraded!!!
class ExpPlotter:

    def __init__(self, app, expPlotData):
        self.app = app
        self.plotTTLChannels = []
        self.plotDACChannels = []
        self.plotDDSChannels = []

    def setTTLChannels(plotTTLChannels):
        self.plotTTLChannels = plotTTLChannels

    def setTTLChannels(plotTTLChannels):
        self.plotTTLChannels = plotTTLChannels
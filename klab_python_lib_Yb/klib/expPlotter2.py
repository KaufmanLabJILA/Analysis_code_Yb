from .imports import *

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
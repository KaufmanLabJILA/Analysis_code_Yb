from .imports import *

class ExpPlotData:
    def __init__(self, exp):
        self.exp = exp
        self.variables = self.getVariables()
        self.lines = self.getMasterScriptLines()
        self.TTLChannels = self.getTTLChannels()
        self.DACChannels = self.getDACChannels()
        self.DDSFreqChannels, self.DDSAmpChannels = self.getDDSChannels()
        self.snapshots = []
        self.getSnapshots()
        self.DACData = self.getDACData()
        self.DDSFreqData = self.getDDSFreqData()
        self.DDSAmpData = self.getDDSAmpData()
        self.TTLData = self.getTTLData()
        self.key_names = self.getKeyNames(exp.key_name)
        self.key = exp.key
        self.pics = exp.pics
        self.reps = exp.reps
        self.experiment_time = exp.experiment_time
        self.experiment_date = exp.experiment_date
        self.data_addr = exp.data_addr
        self.file_id = exp.file_id

    def getKeyNames(self, key_name):
        if (type(key_name) == list):
            return key_name
        elif (type(key_name) == str):
            return [key_name]
        
    def getMasterScriptLines(self):
        mscript = self.exp.f['Master-Parameters']['Master-Script']
        lines = []
        line = []
        for x in mscript:
            c = x.decode('UTF-8')
            if (c == '\n'):
                lines.append("".join(line))
                line = []
            else:
                line += c
        linesUnc = list(map(self.cleanLines, lines))
        linesFil = list(filter(self.filterEmptyLines, linesUnc))
        linesEvalDAC = list(map(self.evaluateDACVariables, linesFil))
        linesEvalDDS = list(map(self.evaluateDDSVariables, linesEvalDAC))
        linesEvalTime = list(map(self.evaluateTimeVariables, linesEvalDDS))
        linesEvalRepeats = self.evaluateRepeats(linesEvalTime)
        linesEvalPulses = self.evaluatePulses(linesEvalRepeats)
        return linesEvalPulses
    
    def getVariables(self):
        variables = {}
        for key, val in self.exp.f['Master-Parameters']['Variables'].items():
            variables[key] = val[0]
        return variables
        
    def cleanLines(self, line):
        line = line.split('%')[0]
        line = line.split('\r')[0]
        return line

    def filterEmptyLines(self, line):
        if (line == ''):
            return False
        else:
            return True
    
    def evaluateRepeats(self, lines):
        repeatLinesAll = []
        inds = [-1]
        repeatLine = -1
        endLine = -1
        for i, line in enumerate(lines):
            repeatMatch = re.search("^repeat:\s+[0-9]+", line)
            endMatch = re.search("^end\s*", line)
            if (repeatMatch):
                lineSplit = re.split('\s+', repeatMatch[0])
                numRepeats = int(lineSplit[1])
                repeatLine = i
            if (endMatch and repeatLine != -1):
                endLine = i
            if (endLine != -1):
                repeatLines = lines[repeatLine+1:endLine]
                repeatLines = repeatLines*numRepeats
                repeatLinesAll.append(repeatLines)
                inds.append(repeatLine)
                inds.append(endLine)
                endLine = -1
                repeatLine = -1
        linesPieces = [lines[inds[2*j]+1:inds[2*j+1]] + repeatLinesAll[j] for j in range(len(repeatLinesAll))] + [lines[inds[-1]+1:]]
        lines = [line for linesPiece in linesPieces for line in linesPiece]
        return lines
        
    def evaluateDACVariables(self, line):
        dacMatch = re.search("^dac:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        dacRampMatch = re.search("^dacramp:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)\*\/]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        if (dacMatch):
            lineSplit = re.split('\s+', dacMatch[0])
            startSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
            for var, val in self.variables.items():
                startSplit = [re.sub("^"+var+"$", str(val), s) for s in startSplit]
            lineSplit[2] = "".join(startSplit)
            lineSplit[2] = str(eval(lineSplit[2]))
            line = " ".join(lineSplit)
        elif (dacRampMatch):
            lineSplit = re.split('\s+', dacRampMatch[0])
            startSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
            endSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[3])
            rampSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[4])
            for var, val in self.variables.items():
                startSplit = [re.sub("^"+var+"$", str(val), s) for s in startSplit]
                endSplit = [re.sub("^"+var+"$", str(val), s) for s in endSplit]
                rampSplit = [re.sub("^"+var+"$", str(val), s) for s in rampSplit]
            lineSplit[2] = "".join(startSplit)
            lineSplit[3] = "".join(endSplit)
            lineSplit[4] = "".join(rampSplit)
            lineSplit[2] = str(eval(lineSplit[2]))
            lineSplit[3] = str(eval(lineSplit[3]))
            lineSplit[4] = str(eval(lineSplit[4]))
            line = " ".join(lineSplit)
        return line
    
    def evaluateDDSVariables(self, line):
        ddsampMatch = re.search("^ddsamp:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        ddsfreqMatch = re.search("^ddsfreq:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        ddsRampAmpMatch = re.search("^ddsrampamp:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)\*\/]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        ddsRampFreqMatch = re.search("^ddsrampfreq:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)\*\/]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        if (ddsampMatch or ddsfreqMatch):
            if (ddsampMatch):
                ddsMatch = ddsampMatch
            else:
                ddsMatch = ddsfreqMatch
            lineSplit = re.split('\s+', ddsMatch[0])
            startSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
            for var, val in self.variables.items():
                startSplit = [re.sub("^"+var+"$", str(val), s) for s in startSplit]
            lineSplit[2] = "".join(startSplit)
            lineSplit[2] = str(eval(lineSplit[2]))
            line = " ".join(lineSplit)
        elif (ddsRampAmpMatch or ddsRampFreqMatch):
            if (ddsRampAmpMatch):
                ddsRampMatch = ddsRampAmpMatch
            else:
                ddsRampMatch = ddsRampFreqMatch
            lineSplit = re.split('\s+', ddsRampMatch[0])
            startSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
            endSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[3])
            rampSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[4])
            for var, val in self.variables.items():
                startSplit = [re.sub("^"+var+"$", str(val), s) for s in startSplit]
                endSplit = [re.sub("^"+var+"$", str(val), s) for s in endSplit]
                rampSplit = [re.sub("^"+var+"$", str(val), s) for s in rampSplit]
            lineSplit[2] = "".join(startSplit)
            lineSplit[3] = "".join(endSplit)
            lineSplit[4] = "".join(rampSplit)
            lineSplit[2] = str(eval(lineSplit[2]))
            lineSplit[3] = str(eval(lineSplit[3]))
            lineSplit[4] = str(eval(lineSplit[4]))
            line = " ".join(lineSplit)
        return line
    
    def evaluateTimeVariables(self, line):
        tMatch = re.search("^t\s+[\+=]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        if (tMatch):
            lineSplit = re.split('\s+', tMatch[0])
            timeSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
            for var, val in self.variables.items():
                timeSplit = [re.sub("^"+var+"$", str(val), s) for s in timeSplit]
            lineSplit[2] = "".join(timeSplit)
            lineSplit[2] = str(eval(lineSplit[2]))
            line = " ".join(lineSplit)
        return line
    
    def evaluatePulses(self, lines):
        pulseLinesAll = []
        inds = [-1]
        for i, line in enumerate(lines):
            pulseonMatch = re.search("^pulseon:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
            if (pulseonMatch):
                lineSplit = re.split('\s+', pulseonMatch[0])
                timeSplit = re.split('(\+|\-|\*|\/|\(|\))', lineSplit[2])
                for var, val in self.variables.items():
                    timeSplit = [re.sub("^"+var+"$", str(val), s) for s in timeSplit]
                lineSplit[2] = "".join(timeSplit)
                pulseTime = str(eval(lineSplit[2]))
                pulseLines = ['on: '+lineSplit[1], 't += '+pulseTime, 'off: '+lineSplit[1]]
                pulseLinesAll.append(pulseLines)
                inds.append(i)
        linesPieces = [lines[inds[j]+1:inds[j+1]] + pulseLinesAll[j] for j in range(len(pulseLinesAll))] + [lines[inds[-1]+1:]]
        lines = [line for linesPiece in linesPieces for line in linesPiece]
        return lines

    def getTTLChannels(self):
        TTLChannels = []
        for line in self.lines:
            channel, _ = self.matchTTL(line)
            if (channel):
                if channel not in TTLChannels:
                    TTLChannels.append(channel)
        return TTLChannels
    
    def getDACChannels(self):
        DACChannels = []
        for line in self.lines:
            channel, _ = self.matchDAC(line)
            if (channel):
                if channel not in DACChannels:
                    DACChannels.append(channel)
        return DACChannels
    
    def getSnapshots(self):
        TTLs, DACs, DDSFreqs, DDSAmps = {}, {}, {}, {}
        for channel in self.TTLChannels:
            TTLs[channel] = 0
        for channel in self.DACChannels:
            DACs[channel] = {'start': 0, 'end': 0, 'ramptime': 0}
        for channel in self.DDSFreqChannels:
            DDSFreqs[channel] = {'start': 0, 'end': 0, 'ramptime': 0}
        for channel in self.DDSAmpChannels:
            DDSAmps[channel] = {'start': 0, 'end': 0, 'ramptime': 0}
        t = 0
        for j, line in enumerate(self.lines):
            ttl, value = self.matchTTL(line)
            if (ttl):
                TTLs[ttl] = value
            dac, DACValues = self.matchDAC(line)
            ddsFreq, DDSFreqValues = self.matchDDSFreq(line)
            ddsAmp, DDSAmpValues = self.matchDDSAmp(line)
            if (dac):
                DACs[dac] = DACValues
            if (ddsFreq):
                DDSFreqs[ddsFreq] = DDSFreqValues
            if (ddsAmp):
                DDSAmps[ddsAmp] = DDSAmpValues
            tMatch = re.search("^t\s+[\+=]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
            if (tMatch):
                newSnapshot = Snapshot(t, copy.deepcopy(TTLs), copy.deepcopy(DACs),
                                      copy.deepcopy(DDSFreqs), copy.deepcopy(DDSAmps))
                self.snapshots.append(newSnapshot)
                lineSplit = re.split('\s+', tMatch[0])
                if (lineSplit[1] == '='):
                    t = float(lineSplit[2])
                elif (lineSplit[1] == '+='):
                    t += float(lineSplit[2])
                for dac in self.DACChannels:
                    if (DACs[dac]['ramptime'] != 0):
                        DACs[dac]['start'] = DACs[dac]['end']
                    DACs[dac]['end'] = 0
                    DACs[dac]['ramptime'] = 0
                for ddsFreq in self.DDSFreqChannels:
                    if (DDSFreqs[ddsFreq]['ramptime'] != 0):
                        DDSFreqs[ddsFreq]['start'] = DDSFreqs[ddsFreq]['end']
                    DDSFreqs[ddsFreq]['end'] = 0
                    DDSFreqs[ddsFreq]['ramptime'] = 0
                for ddsAmp in self.DDSAmpChannels:
                    if (DDSAmps[ddsAmp]['ramptime'] != 0):
                        DDSAmps[ddsAmp]['start'] = DDSAmps[ddsAmp]['end']
                    DDSAmps[ddsAmp]['end'] = 0
                    DDSAmps[ddsAmp]['ramptime'] = 0
        
                    
    def matchTTL(self, line):
        onMatch = re.search("^on:\s+[\-a-zA-Z0-9_]+", line)
        offMatch = re.search("^off:\s+[\-a-zA-Z0-9_]+", line)
        channel = ''
        value = 0
        if (onMatch):
            channel = onMatch[0].split(': ')[1]
            value = 1
        elif (offMatch):
            channel = offMatch[0].split(': ')[1]
            value = 0
        return channel, value
    
    def matchDAC(self, line):
        dacMatch = re.search("^dac: [\-a-zA-Z0-9_]+ [\-a-zA-Z0-9_\.]+", line)
        dacRampMatch = re.search("^dacramp: [\-a-zA-Z0-9_]+ [\-a-zA-Z0-9_\.]+ [\-a-zA-Z0-9_\.]+ [\-a-zA-Z0-9_\.]+", line)
        channel = ''
        values = {'start': 0, 'end': 0, 'ramptime': 0}
        if (dacMatch):
            dacMatchSplit = dacMatch[0].split(' ')
            channel = dacMatchSplit[1]
            values['start'] = float(dacMatchSplit[2])
            values['end'] = 0
            values['ramptime'] = 0
        elif (dacRampMatch):
            dacRampMatchSplit = dacRampMatch[0].split(' ')
            channel = dacRampMatchSplit[1]
            values['start'] = float(dacRampMatchSplit[2])
            values['end'] = float(dacRampMatchSplit[3])
            values['ramptime'] = float(dacRampMatchSplit[4])
        return channel, values
    
    def getDACData(self):
        DACData = {}
        for channel in self.DACChannels:
            ts, DACValues = self.getDACChannelValues(self.snapshots, channel)
            DACData[channel] = pd.DataFrame(np.transpose([ts,DACValues]), columns = ['time', 'value'])
        return DACData
    
    def getTTLData(self):
        ts = arr([s.t for s in self.snapshots])
        ts = np.repeat(ts,2)[1:]
        data = []
        data.append(ts)
        for ttl in self.TTLChannels:
            ttlValues = arr([s.TTLChannels[ttl] for s in self.snapshots])
            ttlValues = np.repeat(ttlValues, 2)[:-1]
            data.append(ttlValues)
        data = np.transpose(data)
        TTLData = pd.DataFrame(data, columns = ['time']+[channel for channel in self.TTLChannels])
        return TTLData
    
    def getDACChannelValues(self, snapshots, channel):
        DACValues = [snapshots[0].DACChannels[channel]['start']]
        ts = [0]
        for s in snapshots[1:]:
            cValues = s.DACChannels[channel]
            rt = cValues['ramptime']
            channelChanged = False
            if (DACValues[-1] != cValues['start'] or rt != 0):
                channelChanged = True
            if (channelChanged):
                ts.append(s.t)
                # add point with values before update
                ts.append(s.t)
                DACValues.append(DACValues[-1])
                DACValues.append(cValues['start'])

                if (rt != 0):
                    ts.append(ts[-1]+rt)
                    DACValues.append(cValues['end'])
        if (ts[-1] < snapshots[-1].t):
            DACValues.append(DACValues[-1])
            ts.append(snapshots[-1].t) 

        return ts, DACValues
    
    def getDDSChannels(self):
        DDSFreqChannels = []
        DDSAmpChannels = []
        for line in self.lines:
            channelFreq, _ = self.matchDDSFreq(line)
            channelAmp, _ = self.matchDDSAmp(line)
            if (channelFreq):
                if channelFreq not in DDSFreqChannels:
                    DDSFreqChannels.append(channelFreq)
            if (channelAmp):
                if channelAmp not in DDSAmpChannels:
                    DDSAmpChannels.append(channelAmp)
        return DDSFreqChannels, DDSAmpChannels
    
    def matchDDSFreq(self, line):
        ddsfreqMatch = re.search("^ddsfreq:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        ddsRampFreqMatch = re.search("^ddsrampfreq:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)\*\/]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        channel = ''
        values = {'start': 0, 'end': 0, 'ramptime': 0}
        if (ddsfreqMatch):
            ddsMatchSplit = ddsfreqMatch[0].split(' ')
            channel = ddsMatchSplit[1]
            values['start'] = float(ddsMatchSplit[2])
            values['end'] = 0
            values['ramptime'] = 0
        elif (ddsRampFreqMatch):
            ddsRampMatchSplit = ddsRampFreqMatch[0].split(' ')
            channel = ddsRampMatchSplit[1]
            values['start'] = float(ddsRampMatchSplit[2])
            values['end'] = float(ddsRampMatchSplit[3])
            values['ramptime'] = float(ddsRampMatchSplit[4])
        return channel, values
    
    def matchDDSAmp(self, line):
        ddsampMatch = re.search("^ddsamp:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        ddsRampAmpMatch = re.search("^ddsrampamp:\s+[\-a-zA-Z0-9_]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)\*\/]+\s+[\+\-\*\/a-zA-Z0-9_\.\(\)]+", line)
        channel = ''
        values = {'start': 0, 'end': 0, 'ramptime': 0}
        if (ddsampMatch):
            ddsMatchSplit = ddsampMatch[0].split(' ')
            channel = ddsMatchSplit[1]
            values['start'] = float(ddsMatchSplit[2])
            values['end'] = 0
            values['ramptime'] = 0
        elif (ddsRampAmpMatch):
            ddsRampMatchSplit = ddsRampAmpMatch[0].split(' ')
            channel = ddsRampMatchSplit[1]
            values['start'] = float(ddsRampMatchSplit[2])
            values['end'] = float(ddsRampMatchSplit[3])
            values['ramptime'] = float(ddsRampMatchSplit[4])
        return channel, values

    def getDDSFreqData(self):
        DDSFreqData = {}
        for channel in self.DDSFreqChannels:
            ts, DDSValues = self.getDDSChannelFreqValues(self.snapshots, channel)
            DDSFreqData[channel] = pd.DataFrame(np.transpose([ts,DDSValues]), columns = ['time', 'value'])
        return DDSFreqData

    def getDDSChannelFreqValues(self, snapshots, channel):
        DDSFreqValues = [snapshots[0].DDSFreqChannels[channel]['start']]
        ts = [0]
        for s in snapshots[1:]:
            cValues = s.DDSFreqChannels[channel]
            rt = cValues['ramptime']
            channelChanged = False
            if (DDSFreqValues[-1] != cValues['start'] or rt != 0):
                channelChanged = True
            if (channelChanged):
                ts.append(s.t)
                # add point with values before update
                ts.append(s.t)
                DDSFreqValues.append(DDSFreqValues[-1])
                DDSFreqValues.append(cValues['start'])

                if (rt != 0):
                    ts.append(ts[-1]+rt)
                    DDSFreqValues.append(cValues['end'])
        if (ts[-1] < snapshots[-1].t):
            DDSFreqValues.append(DDSFreqValues[-1])
            ts.append(snapshots[-1].t) 

        return ts, DDSFreqValues

    def getDDSAmpData(self):
        DDSAmpData = {}
        for channel in self.DDSAmpChannels:
            ts, DDSValues = self.getDDSChannelAmpValues(self.snapshots, channel)
            DDSAmpData[channel] = pd.DataFrame(np.transpose([ts,DDSValues]), columns = ['time', 'value'])
        return DDSAmpData

    def getDDSChannelAmpValues(self, snapshots, channel):
        DDSAmpValues = [snapshots[0].DDSAmpChannels[channel]['start']]
        ts = [0]
        for s in snapshots[1:]:
            cValues = s.DDSAmpChannels[channel]
            rt = cValues['ramptime']
            channelChanged = False
            if (DDSAmpValues[-1] != cValues['start'] or rt != 0):
                channelChanged = True
            if (channelChanged):
                ts.append(s.t)
                # add point with values before update
                ts.append(s.t)
                DDSAmpValues.append(DDSAmpValues[-1])
                DDSAmpValues.append(cValues['start'])

                if (rt != 0):
                    ts.append(ts[-1]+rt)
                    DDSAmpValues.append(cValues['end'])
        if (ts[-1] < snapshots[-1].t):
            DDSAmpValues.append(DDSAmpValues[-1])
            ts.append(snapshots[-1].t) 

        return ts, DDSAmpValues


class Snapshot:
    def __init__(self, t, TTLChannels, DACChannels, DDSFreqChannels, DDSAmpChannels):
        self.t = t
        self.TTLChannels = TTLChannels
        self.DACChannels = DACChannels
        self.DDSFreqChannels = DDSFreqChannels
        self.DDSAmpChannels = DDSAmpChannels
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import nn_spikes

def dataPreProcess(df, spikeLocations=pd.DataFrame([]), threshold=0.85, submission=False, waveformSignalType='original'):
    try:
        assert spikeLocations.shape[0] != 0
        data = joinSpikes(df, spikeLocations)
        knownSpikeIndexes = data[data['knownSpike'] == True].index
        data = getSpikeWaveforms(knownSpikeIndexes, data)
    except AssertionError:
        data = df

    data.insert(len(data.columns), 'predictedSpike', False)
    data.insert(len(data.columns), 'predictedClass', 0)

    data['signalFiltered'] = bandPassFilter(data['signal'])

    data, predictedSpikeIndexes = detectPeaks(data, threshold=threshold)
    data = getSpikeWaveforms(predictedSpikeIndexes, data, waveformSignalType=waveformSignalType)

    data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

    if submission:
        return data, predictedSpikeIndexes
    else:
        return data_training, data_validation, spikeIndexes_training, spikeIndexes_validation

def joinSpikes(data, spikes):

    if not 'knownSpike' in data.columns:
        # Create 2 new columns for spike data and prepare 2 additional columns for predicted spike data
        data.insert(len(data.columns), 'knownSpike', False)
        data.insert(len(data.columns), 'knownClass', 0)

        # Store spike data in new columns
        data.loc[spikes['index'], 'knownSpike'] = True
        data.loc[spikes['index'], 'knownClass'] = spikes['class'].values
    else:
        print("Known spikes already included, returning with original data.")

    return data

def splitData(data, spikeIndexes, trainingShare=0.8):

    # Get split index
    splitIndex = int(data.shape[0] * trainingShare)

    data_training = data.iloc[:splitIndex]
    data_validation = data.iloc[splitIndex:]

    lastTrainingSpike = spikeIndexes[spikeIndexes < len(data_training)][-1]
    spikeSplitIndex = list(spikeIndexes).index(lastTrainingSpike) + 1

    spikeIndexes_training = spikeIndexes[:spikeSplitIndex]
    spikeIndexes_validation = spikeIndexes[spikeSplitIndex:]

    # Return training and validation data
    return data_training, data_validation, spikeIndexes_training, spikeIndexes_validation

def bandPassFilter(signal, lowCut=300.00, highCut=3000.00, sampleRate=25000, order=1):
    # TODO: Calculate something
    nyq = 0.5 * sampleRate
    low = lowCut / nyq
    high = highCut / nyq

    # Generate filter coefficients for butterworth filter
    b, a = butter(order, [low, high], btype='bandpass')

    signalFiltered = lfilter(b, a, signal)
    return signalFiltered

def detectPeaks(data, threshold=0.85):
    df = data.loc[data['signalFiltered'] > threshold]

    peaks = df[(df['signalFiltered'].shift(1) < df['signalFiltered']) &
               (df['signalFiltered'].shift(-1) < df['signalFiltered'])]

    # Insert column to store predicted spike info
    if 'predictedSpike' not in data.columns:
        data.insert(len(data.columns), 'predictedSpike', False)
    if 'predictedClass' not in data.columns:
        data.insert(len(data.columns), 'predictedClass', 0)

    # Create series of
    s = pd.Series(peaks.index)

    doubleCounts = s.loc[s-s.shift(1)<15]

    # Drop indexes of detected peaks if they occur within 15 points of another peak and store as spike indexes
    # Detected spike indexes are shifted by X points to align with labeled dataset used during training and improve similarity to waveforms expected by MLP model
    spikeIndexes = peaks.index.drop(labels=doubleCounts) - 8

    # Truncate negative indexes to zero (used for submission dataset)
    if len(spikeIndexes[spikeIndexes < 1]) > 0:
        # Cast to series to make mutable
        spikeIndexes = spikeIndexes.to_series()
        # Truncate negative indexes to 1. This is the case as in the submission dataset with more noise it is likely
        # that a spike will be detected within the first 8 points.
        spikeIndexes[spikeIndexes < 1] = 1
        print("Truncated index of {} detected peaks to one.".format(len(spikeIndexes[spikeIndexes == 1])))

    data.loc[spikeIndexes, 'predictedSpike'] = True

    print("{} peaks detected.".format(len(spikeIndexes)))

    return data, spikeIndexes.values

def getSpikeWaveforms(peakIndexes, data, window=60, waveformSignalType='original'):
    """

    :param waveformSignalType:
    :param peakIndexes:
    :param data:
    :param window:
    :return:
    """

    # Insert column to store putative spike waveform data
    if 'waveform' not in data.columns:
        data.insert(len(data.columns), 'waveform', None)

    # Iterate over each detected spike
    for index in peakIndexes:

        if 'original' in waveformSignalType:
            # Retrieve a window of signal values
            waveform = data.loc[index - int(window / 4):index + int(3 / 4 * window), 'signal'].tolist()
        elif 'filtered' in waveformSignalType:
            # Retrieve a window of filtered signal values
            waveform = data.loc[index - int(window / 4):index + int(3 / 4 * window), 'signalFiltered'].tolist()
        else:
            raise ValueError("Specify either original or filtered signal to get waveform. Nothing else exists.")

        # Filter waveform to make it less noisy
        # TODO: Review this filter please.
        waveformSmooth = bandPassFilter(waveform)

        # Store waveform values in list
        data.at[index, 'waveform'] = waveformSmooth

    return data


def plotSpikes(signals, spikes):
    """
    Function used to plot detected spikes against signal. Can be used to compare predicted vs. truths.
    :param signals: pandas series containing original signal data
    :param spikes: list of pandas series containing spike locations dtype=Bool
    :return: None
    """
    assert isinstance(spikes, list) and isinstance(signals, list)
    assert isinstance(signals[0], pd.Series) and isinstance(spikes[0], pd.Series)

    fig = go.Figure()

    for signal in signals:
        fig.add_trace(go.Scatter(
            x=signal.index,
            y=signal,
            mode='lines',
            name=signal.name,
            opacity=0.5,
        ))

    for spikeGroup in spikes:

        spikeLocations = spikeGroup[spikeGroup == True].index

        fig.add_trace(go.Scatter(
            x=spikeLocations,
            y=[signals[0][j] for j in spikeLocations],
            mode='markers',
            marker=dict(
                size=8,
                symbol='cross'
            ),
            name=spikeGroup.name
        ))

    fig.show()

def classifySpikesMLP(waveforms, nn):

    # Ensure data is of type pandas dataframe
    assert isinstance(waveforms, pd.Series)

    # Create an empty string to accumulate the count of correct predictions
    predictions = []

    # Iterate over each row in the data
    for waveform in waveforms:
        inputs = np.array(waveform.tolist(), ndmin=2).T

        # Query the network
        outputs = nn.query(inputs)

        # Identify predicted label
        prediction = np.argmax(outputs)

        # Correct label predicted to account for non-zero counting of neuron types and append to list of classified action potentials
        predictions.append(prediction+1)

    return predictions


def getAverageWaveforms(data_training, spikeIndexes_training):
    for index in spikeIndexes_training:
        _, _, label = nn_spikes.getInputsAndTargets(data_training.loc[index, 'waveform'], 4,
                                          data_training.loc[index - 10:index + 5, 'knownClass'])
        data_training.loc[index, 'knownClass'] = label + 1

    detectedSpikes = data_training.loc[spikeIndexes_training]
    class1 = detectedSpikes[detectedSpikes['knownClass'] == 1]['waveform']
    class2 = detectedSpikes[detectedSpikes['knownClass'] == 2]['waveform']
    class3 = detectedSpikes[detectedSpikes['knownClass'] == 3]['waveform']
    class4 = detectedSpikes[detectedSpikes['knownClass'] == 4]['waveform']

    for classWaveforms in [class1, class2, class3, class4]:
        stack = np.vstack(classWaveforms.values)
        np.average(stack[:, 0])

        avgs = []

        for col in range(stack.shape[1]):
            colAvg = np.average(stack[:, col])
            avgs.append(colAvg)

        class4 = pd.Series([avgs]).append(class4)

    fig = go.Figure()

    for trace in class4[1:]:
        fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                                 y=trace,
                                 mode='lines',
                                 line=dict(color='black'),
                                 opacity=0.1,
                                 ))

    fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                             y=class4[0],
                             mode='lines', ))

    fig.show()
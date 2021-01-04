from scipy.signal import butter, lfilter, savgol_filter
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import nn_spikes

def dataPreProcess(df, spikeLocations=pd.DataFrame([]), threshold=0.85, submission=False, detectPeaksOn='signalSavgolBP', waveformWindow=60, waveformSignalType='signalSavgol'):
    try:
        assert spikeLocations.shape[0] != 0
        data = joinSpikes(df, spikeLocations)
        knownSpikeIndexes = data[data['knownSpike'] == True].index
        data = getSpikeWaveforms(knownSpikeIndexes, data)
    except AssertionError:
        data = df

    data.insert(len(data.columns), 'predictedSpike', False)
    data.insert(len(data.columns), 'predictedClass', 0)

    data['signalSavgol'] = savgol_filter(data['signal'], 17, 2)
    data['signalSavgolBP'] = bandPassFilter(data['signalSavgol'])

    data, predictedSpikeIndexes = detectPeaks(data, detectPeaksOn=detectPeaksOn, threshold=threshold)
    data = getSpikeWaveforms(predictedSpikeIndexes, data, window=waveformWindow, signalType=waveformSignalType)

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

def bandPassFilter(signal, lowCut=300.00, highCut=3000.00, sampleRate=25000, order=1, filterType='band'):

    nyq = 0.5 * sampleRate
    low = lowCut / nyq
    high = highCut / nyq

    if 'band' in filterType:
        # Generate filter coefficients for butterworth filter
        b, a = butter(order, [low, high], btype='bandpass')
    elif 'high' in filterType:
        # Generate filter coefficients for butterworth filter
        b, a = butter(order, low, btype='high')
    else:
        b, a = (None, None)

    signalFiltered = lfilter(b, a, signal)
    return signalFiltered

def detectPeaks(data, detectPeaksOn='signalSavgolBP', threshold=0.85):
    df = data.loc[data[detectPeaksOn] > threshold]

    peaks = df[(df[detectPeaksOn].shift(1) < df[detectPeaksOn]) &
               (df[detectPeaksOn].shift(-1) < df[detectPeaksOn])]

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

def getSpikeWaveforms(peakIndexes, data, window=60, signalType='signalSavgol'):
    """

    :param signalType:
    :param peakIndexes:
    :param data:
    :param window:
    :return:
    """
    assert signalType in data.columns

    # Insert column to store putative spike waveform data
    if 'waveform' not in data.columns:
        data.insert(len(data.columns), 'waveform', None)

    # Iterate over each detected spike
    for index in peakIndexes:

        waveform = data.loc[index - int(window / 4):index + int(3 / 4 * window), signalType]

        # Store waveform values in list
        data.at[index, 'waveform'] = waveform.reset_index(drop=True)

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

        # Query the network: output will start from zero
        outputs = nn.query(inputs)

        # Identify predicted label: predictions
        prediction = np.argmax(outputs)

        # Correct label predicted to account for non-zero counting of neuron types and append to list of classified action potentials
        predictions.append(prediction+1)

    return predictions


def getAverageWaveforms(data_training, spikeIndexes_training, classToPlot=0):
    # Loop over each index of a detected spike
    for index in spikeIndexes_training:
        # Get just the label associated with this spike by comparing it to all possible labels within a window of 10 preceeding and
        # 5 succeeding points of spike index, looking in the labelled data column (given spike locations)
        _, _, label = nn_spikes.getInputsAndTargets(data_training.loc[index, 'waveform'], 4,
                                          data_training.loc[index - 10:index + 5, 'knownClass'])

        # Store label and adjust for non-zero indexing
        data_training.loc[index, 'knownClass'] = label + 1

    # Retrieve a dataframe containing only spike entries and select all whose class is 1, 2, 3... etc. for the waveform column.
    # The result is one dataframe with all the spike waveforms for a given class
    detectedSpikes = data_training.loc[spikeIndexes_training]
    class1 = detectedSpikes[detectedSpikes['knownClass'] == 1]['waveform']
    class2 = detectedSpikes[detectedSpikes['knownClass'] == 2]['waveform']
    class3 = detectedSpikes[detectedSpikes['knownClass'] == 3]['waveform']
    class4 = detectedSpikes[detectedSpikes['knownClass'] == 4]['waveform']

    # Store this dataframes in a dictionary
    classes = {'class1': class1,
               'class2': class2,
               'class3': class3,
               'class4': class4}

    # Loop over each of these dataframes and create a numpy array of waveform values stacked vertically. This is to allow you
    # to calculate the mean of the first points, second points, etc. whilst leveraging vectorisation
    for i in classes.keys():
        # Create vertical stack of all waveform values for that class and take average
        stack = np.vstack(classes[i].values)
        np.average(stack[:, 0])

        # Create new list ready to store average values
        avgs = []

        # Loop over each column in stacked waveform values. This is equivalent to going point by point through the waveforms and taking
        # the averages of values at that point for all waveforms in that class
        for col in range(stack.shape[1]):
            colAvg = np.average(stack[:, col])
            # Store average of that point in a list. List will be of same length that the window is when extracting the waveforms
            avgs.append(colAvg)

        # Store list of averages by casting to a series and appending at start of original store of waveforms
        # (this is to make indexing it straightforward as classes will contain different number of waveforms)
        classes[i] = pd.Series([avgs]).append(classes[i])

    # Create new plotly figure
    fig = go.Figure()
    # Retrieve which class is intended to be plotted
    key = list(classes.keys())[classToPlot]

    # Plot all waveforms on the same figure, with 10% opacity. Then plot the average waveform in full opacity.
    for trace in classes[key][1:]:
        fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                                 y=trace,
                                 mode='lines',
                                 line=dict(color='black'),
                                 opacity=0.1,
                                 ))

    fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                             y=classes[key][0],
                             mode='lines', ))

    fig.show()
from scipy.signal import butter, lfilter, savgol_filter
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import nn_spikes

def dataPreProcess(df, spikeLocations, threshold=0.85, submission=False, detectPeaksOn='signalSavgolBP', waveformWindow=60, waveformSignalType='signalSavgol'):

    data = df

    data['signalSavgol'] = savgol_filter(data['signal'], 17, 2)
    data['signalSavgolBP'] = bandPassFilter(data['signalSavgol'])

    data, predictedSpikeIndexes = detectPeaks(data, detectPeaksOn=detectPeaksOn, threshold=threshold)
    data = getSpikeWaveforms(predictedSpikeIndexes, data, window=waveformWindow, signalType=waveformSignalType)

    if submission:
        print("Returning with {} detected spikes.".format(len(predictedSpikeIndexes)))
        return data, predictedSpikeIndexes
    else:
        data = joinKnownSpikeClasses(data, spikeLocations)
        # Assign known labels and drop any detected spikes that refer to more than one label
        data, predictedSpikeIndexes = assignKnownClassesToDetectedSpikes(data, predictedSpikeIndexes)

        data_training, data_validation, spikeIndexes_training, spikeIndexes_validation = splitData(data, predictedSpikeIndexes)

        print("Returning with {} detected spikes.".format(len(predictedSpikeIndexes)))
        return data_training, data_validation, spikeIndexes_training, spikeIndexes_validation


def joinKnownSpikeClasses(data, spikes):

    if not 'knownSpike' in data.columns:
        # Create 2 new columns for spike data and prepare 2 additional columns for predicted spike data
        data.insert(len(data.columns), 'knownSpike', False)
        data.insert(len(data.columns), 'knownClass', None)

        # Store spike data in new columns
        data.loc[spikes['index'], 'knownSpike'] = True
        data.loc[spikes['index'], 'knownClass'] = spikes['class'].values - 1 # adjustment of classes to count from zero
    else:
        print("Known spikes already included, returning with original data.")

    return data

def assignKnownClassesToDetectedSpikes(data, predictedSpikeIndexes):
    print("Assigning known classes to detected spikes.")
    duffLabels = []

    data.insert(len(data.columns), 'assignedKnownClass', None)

    # Retrieve non-zero spike labels from list of known spike labels in window either side of detected spike
    for index in predictedSpikeIndexes:
        knownClasses = data.loc[index - 10:index + 5, 'knownClass']
        possibleLabels = knownClasses[4:-2][knownClasses != 0].values

        # If no non-zero spike labels are detected, extend window range and try again
        if len(possibleLabels) == 0:
            possibleLabels = knownClasses[knownClasses != 0].values
            # If still no non-zero spike labels are detected, raise an error because the spike detected could be a false positive
            if len(possibleLabels) == 0:
                raise Warning(
                    "No labels detectable for detected spike with index {}. label window: {}".format(knownClasses.index,
                                                                                                     knownClasses))

        # If more than one non-zero spike labels are detected within the window, assert they are all the same, raise error if not
        if len(possibleLabels) > 1:
            try:
                assert (possibleLabels[0] == possibleLabels).all()
            except AssertionError:
                # More than two knownClass labels for a single spike found at indexes: 54412, 87433, 165493, 232479, 299250, 312319, 339791, 472193, 980407
                data.loc[index, 'assignedKnownClass'] = 666
                duffLabels.append(index)
                continue

        # Retrieve the target label and account for non-zero count
        data.loc[index, 'assignedKnownClass'] = possibleLabels[0]

    # Drop all detected peaks that resemble spikes labelled with two different labels as these may cause inaccuracies in the model training
    if len(duffLabels) > 0:
        # Use list comprehension to retrieve indexes of duff labels within list of predicted spike indexes
        duffLocations = [np.where(predictedSpikeIndexes == a)[0][0] for a in duffLabels]
        # Create np array of Trues of same shape as list of predicted spike indexes
        mask = np.ones(len(predictedSpikeIndexes), dtype=bool)
        # Set all elements at those indexes to false
        mask[duffLocations] = False
        # Filter the list of predicted spike indexes to drop the spikes with duff labels
        predictedSpikeIndexes = predictedSpikeIndexes[mask]
        print("Dropped {} detected spikes that relate to more than one label. (Indexes: {})".format(len(duffLocations),
                                                                                                    duffLabels))

    return data, predictedSpikeIndexes

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

    # Insert columns to store predicted spike info
    data.insert(len(data.columns), 'predictedSpike', False)
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

    # Retrieve a dataframe containing only spike entries and select all whose class is 1, 2, 3... etc. for the waveform column.
    # The result is one dataframe with all the spike waveforms for a given class
    detectedSpikes = data_training.loc[spikeIndexes_training]
    classWaveforms = detectedSpikes[detectedSpikes['assignedKnownClass'] == classToPlot]['waveform']

    # Create vertical stack of all waveform values for that class and take average
    stack = np.vstack(classWaveforms.values)
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
    classWaveforms = pd.Series([avgs]).append(classWaveforms)

    # Create new plotly figure
    fig = go.Figure()

    # Plot all waveforms on the same figure, with 10% opacity. Then plot the average waveform in full opacity.
    for trace in classWaveforms[1:]:
        fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                                 y=trace,
                                 mode='lines',
                                 line=dict(color='black'),
                                 opacity=0.1,
                                 ))

    fig.add_trace(go.Scatter(x=np.linspace(0, 100, 101),
                             y=classWaveforms[0],
                             mode='lines', ))

    fig.show()
from scipy.signal import butter, lfilter
import plotly.graph_objects as go
import pandas as pd

def joinSpikes(data, spikes):

    if not 'knownSpike' in data.columns:
        # Create 2 new columns for spike data and prepare 2 additional columns for predicted spike data
        data.insert(len(data.columns), 'knownSpike', False)
        data.insert(len(data.columns), 'knownClass', 0)
        data.insert(len(data.columns), 'predictedSpike', False)
        data.insert(len(data.columns), 'predictedClass', 0)

        # Store spike data in new columns
        data.loc[spikes['index'], 'knownSpike'] = True
        data.loc[spikes['index'], 'knownClass'] = spikes['class'].values
    else:
        print("Spike data already exists, just returning input data.")

    return data


def splitData(data, trainingShare=0.8):

    try:
        assert all(col in data.columns for col in ['knownSpike', 'knownClass', 'predictedSpike', 'predictedClass'])
    except AssertionError:
        raise AssertionError("Prep your data please. Run joinSpikes().")

    # Get split index
    splitIndex = int(data.shape[0] * trainingShare)

    # Return training and validation data
    return data.iloc[:splitIndex], data.iloc[splitIndex:]

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

    doubleCounts = s.loc[s-s.shift(1)<10]

    # Drop indexes of detected peaks if they occur within 10 points of another peak and store as spike indexes
    # Detected spike indexes are shifted by X points to align with labeled dataset used during training and improve similarity to waveforms expected by MLP model
    spikeIndexes = peaks.index.drop(labels=doubleCounts) - 8

    data.loc[spikeIndexes, 'predictedSpike'] = True

    print("{} peaks detected.".format(len(spikeIndexes)))

    return data, spikeIndexes

def getSpikeWaveforms(peakIndexes, data, window=100):
    """

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

        # Retrieve a window of signal values
        waveform = data.loc[index - int(window / 4):index + int(3 / 4 * window), 'signal'].tolist()

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
from scipy.signal import butter, lfilter, savgol_filter
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from nn_spikes import getInputsAndTargets

def dataPreProcess(data, spikeLocations, threshold=0.85, submission=False, detectPeaksOn='signalSavgolBP', waveformWindow=60, waveformSignalType='signalSavgol'):
    """
    Function creates all derived data columns necessary to begin further analyses as a pre-processing step and returns the larger
    data table and predicted spike indexes as a tuple that will be unpacked.
    :param data: original spike signal data
    :param spikeLocations: original spike data from moodle download that contains labelled spike info
    :param threshold: amplitude threshold used to detect spikes
    :param submission: boolean used to indicate if data being passed is submission data and therefore if known labels should not be attached
    :param detectPeaksOn: string indicating which column to detect peaks on. defaults to the savgol + bandpass filtered signal
    :param waveformWindow: size of window used to extract putative spike waveform
    :param waveformSignalType: which signal to extract the spike waveform from. this is to distinguish between the signals
    used for spike detection and for waveform extraction
    :return: return tuple of new data table and list of predicted spike indexes
    """

    # Check if one of the filtered signal columns is present. If not, run signal filtering and insert in new columns
    if not 'signalSavgol' in data.columns:

        # Assert that signal column contains no null or inf values
        assert all(pd.notnull(data['signal']))
        assert all(~np.isinf(data['signal']))

        # Filter the original signal with a Savitzky-Golay filter, and filter the result with a bandpass filter
        data['signalSavgol'] = savgol_filter(data['signal'], 17, 2)                 # Can fail for some Windows builds: https://github.com/matplotlib/matplotlib/issues/18157
        data['signalSavgolBP'] = bandPassFilter(data['signalSavgol'])
        data['signalHP'] = bandPassFilter(data['signal'], lowCut=100, filterType='high')
        data['signalHPSavgol'] = savgol_filter(data['signalHP'], 17, 2)

        # Detect peaks in provided signal, returned as index positions
        data, predictedSpikeIndexes = detectPeaks(data, detectPeaksOn=detectPeaksOn, threshold=threshold)
    else:
        # If filtered signal columns are present, retrieve predicted spike indexes by filtering column 'predictedSpike'
         predictedSpikeIndexes = data[data['predictedSpike'] == True].index

    # Assert that the correct filtered signal data is present before extracting spike waveforms
    assert waveformSignalType in data.columns
    # Extract the putative spike waveforms at the predicted spike locations and store in new column
    data['waveform'] = getSpikeWaveforms(data.loc[:, waveformSignalType], predictedSpikeIndexes, window=waveformWindow)

    # If using training data, attach the known spike labels to the predicted spike locations
    if not submission:
        # First join the known spike information with the existing data table
        data = joinKnownSpikeClasses(data, spikeLocations)
        # Then assign the known labels to predicted spike locations and drop any detected spikes that refer to more than one label
        data, predictedSpikeIndexes = assignKnownClassesToDetectedSpikes(data, predictedSpikeIndexes)
    else:
        pass

    # Let the homie know wassup and return the pre-processed data and predicted spike locations back to the command centre
    print("Returning with {} detected spikes.".format(len(predictedSpikeIndexes)))
    return data, predictedSpikeIndexes

def joinKnownSpikeClasses(data, spikes):
    """
    Function evaluates if the known spike data has already been added, and adds if not.
    :param data: data table to attach new columns to
    :param spikes: original spike data from moodle download that contains labelled spike info
    :return: return full data table with known spike information included
    """

    # Check for if columns exist already
    if not 'knownSpike' in data.columns:
        # Create 2 new columns for known spike data: known spike class, containing class labels, and known spike for spike locations,
        # indicated by a boolean True when it is a spike
        data.insert(len(data.columns), 'knownSpike', False)
        data.insert(len(data.columns), 'knownClass', -1)                        # negative one used to indicate no spike

        # Store spike data in the new columns. Set all spike checks to True at known spike locations
        data.loc[spikes['index'], 'knownSpike'] = True
        # Store all spike classes in relevant locations.
        # Here we adjust to base-zero, which is to make things easier when identifying predicted classes
        data.loc[spikes['index'], 'knownClass'] = spikes['class'].values - 1    # adjustment of classes to count from zero
    else:
        print("Known spikes already included. Returning with original data.")

    # Return the same data table with the new columns attached if they didn't exist before
    return data

def assignKnownClassesToDetectedSpikes(data, predictedSpikeIndexes):
    """
    Function checks for known labels in window around predicted spike where it would reasonably expect a known spike to be.
    Hard fail here for when the spike is a false positive and no known label is detected, in line with the
    marking criteria on the assessment sheet.
    :param data: data containing all signal, predicted spike information, and other pre-processed data so far
    :param predictedSpikeIndexes: a list-like containing all predicted spike locations
    :return: return the same data table with known class labels shifted, and the list of predicted spike indexes but
    with the spike referring to more than one label removed, so as not to be used in training and therefore obscure the
    learned waveforms. Roughly 30-35 spikes of this kind were regularly detected.
    """

    # Check if column already exists, and if so, return without further processing.
    if 'assignedKnownClass' in data.columns:
        print("Known classes already assigned. Returning with original data.")

    # If not, continue with processing the known spike labels
    else:

        print("Assigning known classes to detected spikes.")
        # Create list to store all duff labels - labels within close proximity to one another, typically on the same rising
        # edge of a single spike, thus rendering it not useful for training purposes.
        duffLabels = []

        # Insert a new columns with default value one at the end of the existing data table
        data.insert(len(data.columns), 'assignedKnownClass', None)

        # Evaluate each predicted spike
        for index in predictedSpikeIndexes:

            # Retrieve a window of values from the known class column, more heavily weighted to trailing values since the
            # peak detection will always result in locations behind the known spike
            knownClassesWindow = data.loc[index - 10:index + 5, 'knownClass']
            # Retrieve spike labels that aren't -1 from list of known spike labels in the a smaller allowable window of indexes
            possibleClasses = knownClassesWindow[4:-2][knownClassesWindow != -1].values

            # If no spike labels are detected (that aren't -1), extend window range and try again. This is to give us the highest
            # chance of detecting just a single spike label at first, although two true spikes occurring within 15 indexes is unlikely
            if len(possibleClasses) == 0:
                possibleClasses = knownClassesWindow[knownClassesWindow != -1].values
                # If still no spike labels are detected, raise an error because the spike detected could be a false positive
                assert len(possibleClasses) != 0, "\nNo labels detectable for detected spike with index {}. \nLabel window: {}\nPossible labels: {}".format(knownClassesWindow.index,knownClassesWindow, possibleClasses)

            # Assert the possible classes we detected are of the data type numpy array
            assert isinstance(possibleClasses, np.ndarray)

            # If more than one spike label is detected within the window, assert they are all the same, and raise error if not
            if len(possibleClasses) > 1:
                try:
                    assert len(np.unique(possibleClasses)) == 1

                except AssertionError:
                    # Note the repeat label of a single spike with the sign of the devil for traceability.
                    data.loc[index, 'assignedKnownClass'] = 666
                    # Add the spike location to the duff labels store, for use in filtering these from the predicted spike list later
                    duffLabels.append(index)
                    continue

            # The resulting list of possible classes should be a single item or repeats of the same unique class. Assign this
            # class to the predicted spike location in a separate column used for training
            data.loc[index, 'assignedKnownClass'] = possibleClasses[0]

        # Drop all detected peaks that resemble spikes labelled with two different labels as these may cause inaccuracies in the model training
        if len(duffLabels) > 0:
            # Use list comprehension to retrieve indexes of duff labels within list of predicted spike indexes
            duffLocations = [np.where(predictedSpikeIndexes == a)[0][0] for a in duffLabels]
            # Create np array of Trues of same shape as list of predicted spike indexes
            mask = np.ones(len(predictedSpikeIndexes), dtype=bool)
            # Set all elements at the locations of the repeated labels to false
            mask[duffLocations] = False
            # Filter the list of predicted spike indexes to drop the spikes with duff labels
            predictedSpikeIndexes = predictedSpikeIndexes[mask]
            print("Dropped {} detected spikes that relate to more than one label. (Indexes: {})".format(len(duffLocations),
                                                                                                        duffLabels))

    # Return the data with known labelled assigned to predicted spikes, and predicted spikes cleaned of spikes associated to duff labels
    return data, predictedSpikeIndexes

def splitData(data, spikeIndexes, trainingShare=0.8):
    """
    Function splits full data table into training and validation parts.
    :param data: data table
    :param spikeIndexes: spike indexes, necessary to also split the spike locations effectively
    :param trainingShare: share used for training, default 80%, with the remainder making up the validation dataset
    :return: returns split data and spike locations
    """

    # Get split index by the integer position of 80% of the data's length
    splitIndex = int(data.shape[0] * trainingShare)

    # Create new training and validation data sets
    data_training = data.iloc[:splitIndex]
    data_validation = data.iloc[splitIndex:]

    # Identify the last spike in the training dataset, and use this to identify the index to split the list of spike locations on
    lastTrainingSpike = spikeIndexes[spikeIndexes < len(data_training)][-1]
    spikeSplitIndex = list(spikeIndexes).index(lastTrainingSpike) + 1

    # Split the spike locations into training and validation sets
    spikeIndexes_training = spikeIndexes[:spikeSplitIndex]
    spikeIndexes_validation = spikeIndexes[spikeSplitIndex:]

    # Return training and validation data
    return data_training, data_validation, spikeIndexes_training, spikeIndexes_validation

def bandPassFilter(signal, lowCut=300.00, highCut=3000.00, sampleRate=25000, order=1, filterType='band'):
    """
    Function to create and apply a bandpass filter using scipy's signal processing library. Uppper and lower frequencies used to
    select the range of frequencies passed by the filter, with frequencies outside this range being blocked.
    :param signal: signal data
    :param lowCut: lower frequency, default 300 used from literature
    :param highCut: upper frequency, default 3000 used from literature
    :param sampleRate: sample rate of signal data, unchanging 25kHz for this assignment
    :param order: order of bandpass filter
    :param filterType: specify the type of filter: highpass, lowpass, bandpass
    :return: filtered signal
    """

    # Calculate the Nyquist frequency as half the sampling rate, and use this to calculate the new low and high frequencies used in the filter
    nyq = 0.5 * sampleRate
    low = lowCut / nyq
    high = highCut / nyq

    # For both bandpass and highpass filters, retrieve the filter coefficients for a Butterworth filter, used in the bandpass filter application
    if 'band' in filterType:
        # Generate filter coefficients for butterworth filter
        b, a = butter(order, [low, high], btype='bandpass')
    elif 'high' in filterType:
        # Generate filter coefficients for butterworth filter
        b, a = butter(order, low, btype='high')
    else:
        b, a = (None, None)

    # Filter the signal by using scipy's lfilter function on the 1D signal data, using the previously calculated coefficients
    signalFiltered = lfilter(b, a, signal)
    return signalFiltered

def detectPeaks(data, detectPeaksOn='signalSavgolBP', threshold=0.85):
    """
    Detect the peaks from within the signal data by first applying an amplitude threshold and then evaluating neighbouring points
    within the spikes to find the local maxima.
    :param data: full data table containing extracellular neuron recordings to detect spikes in
    :param detectPeaksOn: column to detect peaks in, default is the signal filtered with the savgol filter plus bandpass
    :param threshold: amplitude threshold used to separate the signal from the noise
    :return: returns the data table with detected peaks identified, and a list-like of the indexes at which they occurred
    """

    # Retrieve all signal points above the amplitude threshold in the desired signal
    df = data.loc[data[detectPeaksOn] > threshold]

    # Retrieve all peaks by comparing neighbouring points and finding the points higher than both points either side
    peaks = df[(df[detectPeaksOn].shift(1) < df[detectPeaksOn]) &
               (df[detectPeaksOn].shift(-1) < df[detectPeaksOn])]

    # Insert columns into data table to store predicted spike info
    data.insert(len(data.columns), 'predictedSpike', False)
    data.insert(len(data.columns), 'predictedClass', -1)

    # Create series using the predicted peak locations as the index
    s = pd.Series(peaks.index)

    # Retrieve all points that are a repeat within a 15-sample window as double counts of the same peak
    doubleCounts = s.loc[s-s.shift(1)<15]

    # Drop detected peaks if they occur within 15 points of another peak
    # Detected spike indexes are shifted by 8 points to align with labeled dataset, making label assignment easier
    spikeIndexes = peaks.index.drop(labels=doubleCounts) - 8

    # Truncate negative indexes to zero (used for submission dataset where spikes occur within 8 samples of the start)
    if len(spikeIndexes[spikeIndexes < 1]) > 0:
        # Cast index to series to make mutable
        spikeIndexes = spikeIndexes.to_series()
        # Truncate all indexes below 1 to 1
        spikeIndexes[spikeIndexes < 1] = 1
        print("Truncated index of {} detected peaks to one.".format(len(spikeIndexes[spikeIndexes == 1])))

    # Identify predicted spike locations in data table with boolean True
    data.loc[spikeIndexes, 'predictedSpike'] = True

    # Notify how many peaks were detected and return from function the data table and the detected spike locations
    print("{} peaks detected.".format(len(spikeIndexes)))
    return data, spikeIndexes.values

def getSpikeWaveforms(signalData, spikeIndexes, window=60):
    """
    Function extracts a window of signal values around the spike locations.
    :param signalData: signal data
    :param spikeIndexes: spike locations
    :param window: window size denotes the number of samples to extract to contain the putative spike waveform. A smaller
    window size will not mean a lower sampling frequency but a smaller number of points either side of the spike location
    :return: returns a series containing a spike waveform for each spike location with the number of sample points equal
    to the window size plus 1 (due to Python slicing convention)
    """

    # Create an empty series using the same index as the signal data to return the waveforms in a way that can be used to
    # store the values in the full dataframe, and an empty list to collect the waveforms as an intermediary step
    s = pd.Series(index=signalData.index, dtype=object, name='waveform')
    waveforms = []

    # Iterate over each detected spike
    for index in spikeIndexes:

        # Retrieve the window of signal values for that spike and append to list of waveforms, forming a list of series
        waveforms.append(signalData.loc[index - int(window / 4):index + int(3 / 4 * window)].reset_index(drop=True))

    # Store waveform values in a series at each of the corresponding spike locations
    s.loc[spikeIndexes] = pd.Series(waveforms, index=spikeIndexes)

    return s

def plotSpikes(signals, spikes):
    """
    Function used to plot detected spikes against signal. Can be used to compare predicted spikes against known spikes, and
    the performance of alternative filtering results.
    :param signals: list of pandas series containing different signal data
    :param spikes: list of pandas series containing boolean indication of spike locations
    :return: None
    """
    # Assert data types are as expected
    assert isinstance(spikes, list) and isinstance(signals, list)
    assert isinstance(signals[0], pd.Series) and isinstance(spikes[0], pd.Series)

    # Create a plotly graph object figure
    fig = go.Figure()

    # Iterate over each signal data in the signals list and add the trace to the figure
    for signal in signals:
        fig.add_trace(go.Scatter(
            x=signal.index,
            y=signal,
            mode='lines',
            name=signal.name,
            opacity=0.5,
        ))

    # Iterate over each group of spike locations and add as a scatter plot to the figure, with the same x axis
    for spikeGroup in spikes:
        # Retrieve the spike locations from the index of the filtered spike locations data. Only positives (spikes) should be plotted.
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

    # Display the figure containing several plots
    fig.show()

def classifySpikesMLP(waveforms, nn):
    """
    Function returns list of predicted spike classes, based on the extracted spike waveforms
    :param waveforms: pandas series of extracted waveforms
    :param nn: the neural network to be used to predict the classification
    :return: returns list of predicted classes
    """

    # Ensure data is of type pandas series
    assert isinstance(waveforms, pd.Series), "Waveform should be stored as series of series."

    # Create an empty list to store the predictions
    predictions = []

    # Iterate over each waveform and generate the inputs in the same way as when training the network
    for waveform in waveforms:
        inputs, _ = getInputsAndTargets(waveform, 4, 0)

        # Query the network and append results to predictions store
        predictions.append(nn.query(inputs))

    return predictions

def getAverageWaveforms(data_training, spikeIndexes_training, classToPlot=0):
    """
    Function retrieves all waveform extracts for a given class and creates the average of these waveforms by evaluating each
    sample value and finding the mean for that index.
    :param data_training: data table containing all signal, spike and waveform data
    :param spikeIndexes_training: list-like of spike indexes
    :param classToPlot: spike class to generate and plot the average waveform for
    :return: None, but displays a figure
    """

    # Retrieve a dataframe containing only spike entries and select the waveform extracts for a given class
    detectedSpikes = data_training.loc[spikeIndexes_training]
    classWaveforms = detectedSpikes[detectedSpikes['assignedKnownClass'] == classToPlot]['waveform']

    # Create vertical stack of all waveform values for that class
    stack = np.vstack(classWaveforms.values)

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

    # Create new Plotly graph objects figure
    fig = go.Figure()

    # Plot all waveforms on the same figure, with 10% opacity. Then plot the average waveform in full opacity on top.
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
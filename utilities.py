from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import plotly.figure_factory as ff

def getBatchSize(batchSize, lenData):
    """
    Returns batch size used for mini-batch training.
    :param batchSize:
    :param lenData:
    :return:
    """
    if not batchSize:
        return lenData
    else:
        # Only allow batch sizes that don't discard any data (can improve later but low priority)
        assert lenData % batchSize == 0
        pass

def createResultsRepo(hiddenNodes=(200, 500, 700, 900)):
    results = {}
    for depth in hiddenNodes:
        results[str(depth)] = {'trainingCurve': [],
                               'validationCurve': [],
                               'trainingData': None,
                               'validationData': None,
                               'nn': None}

    return results

def plotLearningCurves(results):
    fig = go.Figure()

    for key in results.keys():
        fig.add_trace(go.Scatter(
            y=results[key]['trainingCurve'],
            line=dict(width=1, dash='dash'),
            name=str(key) + ' training'
        ))

        fig.add_trace(go.Scatter(
            y=results[key]['validationCurve'],
            mode='lines',
            name=str(key) + ' validation'
        ))

    fig.show()

def getConfusion(actual, predicted):
    labels = (1, 2, 3, 4)
    np.set_printoptions(precision=2)

    cm = confusion_matrix(actual, predicted, labels=labels)
    cr = classification_report(actual, predicted, labels=labels)

    plotConfusion(cm)
    print(cr)

def plotConfusion(matrix, x=(1, 2, 3, 4), y=(1, 2, 3, 4)):
    # change each element of z to type string for annotations
    matrixText = [[str(y) for y in x] for x in matrix]

    # set up figure
    fig = ff.create_annotated_heatmap(matrix, x=x, y=y, annotation_text=matrixText, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=17),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=17),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()
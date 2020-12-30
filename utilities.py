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
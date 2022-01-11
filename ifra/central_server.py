class CentralServer:
    """Implementation of the notion of central server in federated learning.

    It monitors changes in a given list of remote GCP directories, were nodes are expected to write their models.
    Upon changes of the model files, the central server download them. Each time a new model file is downloaded, the
    central model is updated and sent in another GCP directory. This directory should be monitored by each node.
    """
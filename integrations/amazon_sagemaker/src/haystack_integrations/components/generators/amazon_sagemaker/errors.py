class SagemakerError(Exception):
    """
    Parent class for all exceptions raised by the Sagemaker component
    """


class AWSConfigurationError(SagemakerError):
    """Exception raised when AWS is not configured correctly"""


class SagemakerNotReadyError(SagemakerError):
    """Exception for issues that occur during Sagemaker inference"""


class SagemakerInferenceError(SagemakerError):
    """Exception for issues that occur during Sagemaker inference"""

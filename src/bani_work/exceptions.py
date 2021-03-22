class WrongInputDataException(Exception):
    pass

class LabelDoesNotExists(Exception):
    pass

class WrongGeneratorOutputException(Exception):
    pass

class VectorNotAssignedException(Exception):
    pass

class WrongModelOutputException(Exception):
    pass

class AttemptedUsingEmptyFAQ(Exception):
    pass

class AttemptedUsingNonUsableFAQ(Exception):
    pass


class TrainDataInvalid(Exception):
    pass
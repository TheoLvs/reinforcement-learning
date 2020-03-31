

import uuid


class BaseObject:
    def __init__(self):

        self.id = str(uuid.uuid1())


import attr

@attr.s(slots = True)
class Agent:

    # Agent id
    id = attr.ib()
    id.default
    def _init_id(self):
        return str(uuid.uuid1())


    # Inactive after destroy
    _destroyed = attr.ib(repr = False,default = False)


    def destroy(self):
        self._destroyed = True

    @property
    def destroyed(self):
        return self._destroyed

    @property
    def alive(self):
        return not self._destroyed

    def __getitem__(self,key):
        return self.get(key)


    def get(self,key):
        return getattr(self,key)

    def to_dict(self):
        return attr.asdict(self)

        
    
    
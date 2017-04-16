class State(object):
    """Base class for a state in a state machine."""

    def __init__(self, args = None):
        """
        Initializes the state when it is first switched to.

        Args:
            args: Arguments to initialize the state with.
        """
        pass

    async def update(self, owner):
        """
        Executes the state's behavior for the current tick.

        Args:
            owner: The object to affect behavior for.
        
        Returns:
            If the object's state should be changed, returns the class of the new state.
            Otherwise, return None.
        """
        return None

    def getName(self):
        """
        Gets the name of the state.

        Returns:
            The name of the state as a string.
        """
        return type(self).__name__

class StateMachine(object):
    """Used to keep track of and change an object's state."""

    def __init__(self, owner, args = None):
        """
        Initializes the state machine.

        Args:
            owner: The object that the state machine is controlling.
        """
        self.owner = owner
        self._state = None

    async def update(self):
        """Executes the current state behavior during the current tick."""
        newState = await self._state.update(self.owner)
        if newState != None:
            await self.changeState(newState)

    async def changeState(self, newState):
        """
        Changes the state that the state machine is in.

        Args:
            newState: The state to change to.
        """
        newStateName = newState.getName()
        if self._state:
            print("Old state: " + self._state.getName())
        print("New state: " + newStateName)
        self._state = newState

    def getState(self):
        """
        Gets the name of the current state.

        Returns:
            The name of the current state as a string.
        """
        return self._state.getName()
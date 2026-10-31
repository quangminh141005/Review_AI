class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, function): # add id later
        self.tools[name] = function

    def execute(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        return self.tools[name](**arguments)
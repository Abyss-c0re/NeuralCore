class ConfirmationRequired(Exception):
    def __init__(self, action_name, args, preview=""):
        self.action_name = action_name
        self.args = args
        self.preview = preview or f"Action {action_name}({args})"
        super().__init__(self.preview)

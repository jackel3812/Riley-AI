class RileyCore:
    def __init__(self):
        self.mode = "default"
        self.personality = "friendly"
        self.memory = []

    def set_mode(self, mode):
        self.mode = mode
        return f"Riley mode set to: {mode}"

    def set_personality(self, p):
        self.personality = p
        return f"Riley is now acting like: {p}"

    def think(self, prompt):
        memory_context = "\n".join(self.memory[-5:])
        return f"[Riley Mode: {self.mode} | Personality: {self.personality}]\n{memory_context}\n[User]: {prompt}"

    def remember(self, line):
        self.memory.append(line)
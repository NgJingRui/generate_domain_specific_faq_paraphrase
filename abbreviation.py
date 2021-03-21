class Abbreviation():
    def __init__(self, abbrev: str, expansion: str, index: int):
        self.abbrev = abbrev
        self.expansion = expansion
        self.index = index
        self.start_index = index - len(expansion.split()) - 1
        self.end_index = index + 1

    def print_all(self):
        print(f"self.abbrev -> {self.abbrev}")
        print(f"self.expansion -> {self.expansion}")
        print(f"self.index -> {self.index}")
        print(f"self.start_index -> {self.start_index}")
        print(f"self.end_index -> {self.end_index}")

    def construct_abbrev_with_expansion(self):
        return f"{self.expansion} ({self.abbrev})"
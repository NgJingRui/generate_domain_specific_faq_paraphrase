# Import the necessary packages
from consolemenu import *
from consolemenu.items import *

# Create the menu
menu = ConsoleMenu("Generate Domain-Specific Paraphrases", "Choose your use case below:")

command_1 = CommandItem("Batch Generate",  "python batch_generate_app_interactive.py")
command_2 = CommandItem("Adhoc Generate",  "python adhoc_generate_app_interactive.py")
command_3 = CommandItem("View Similar Questions within FAQ",  "python find_all_similar_questions_within_faq_interactive.py")


# Once we're done creating them, we just add the items to the menu
menu.append_item(command_1)
menu.append_item(command_2)
menu.append_item(command_3)

# Finally, we call show to show the menu and allow the user to interact
menu.show()
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def ascii_art():
    print("\n")                                                                         
    print(color.GREEN + color.BOLD + "   ______________    ____  __    ____________  ________" + color.END) 
    print(color.GREEN + color.BOLD + "  / ___/_  __/   |  / __ \/ /   /  _/ ____/ / / /_  __/" + color.END) 
    print(color.GREEN + color.BOLD + "  \__ \ / / / /| | / /_/ / /    / // / __/ /_/ / / /   " + color.END) 
    print(color.GREEN + color.BOLD + " ___/ // / / ___ |/ _, _/ /____/ // /_/ / __  / / /    " + color.END) 
    print(color.GREEN + color.BOLD + "/____//_/ /_/  |_/_/ |_/_____/___/\____/_/ /_/ /_/     " + color.END)        
    print("\n\nGPU Roofline and Kernel Optimization Tool")
    print("Version: 1.1.6")
    print(color.BOLD+"Authors: A. Zeni, E. Del Sozzo, D. Conficconi, E. D'Arnese"+color.END)
                                                                              

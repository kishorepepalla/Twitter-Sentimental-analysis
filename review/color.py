
from colorama import Fore, Back, Style,init
init()
f='some red text'
print(Fore.RED,f )
print(Back.GREEN + 'and with a green background')
print(Style.DIM + 'and in dim text')
print(Style.RESET_ALL)
print('back to normal now')

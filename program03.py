x = 0
y = 0
m = 4
n = 3
print("Initial State = (0,0)")
print("Capacities = (4,3)")
print("Goal State = (2,y)")
while x != 2:
    r = int(input("Enter the rule: "))
    if r == 1:
        x = m
    elif r == 2:
        y = n
    elif r == 3:
        x = 0
    elif r == 4:
        y = 0
    elif r == 5:
        t = n - y
        y = n
        x -= t
    elif r == 6:
        t = m - x
        x = m
        y -= t
    elif r == 7:
        y += x
        x = 0
    elif r == 8:
        x += y
        y = 0
    else:
        print("Invalid Rule")
        break
print(x, y)

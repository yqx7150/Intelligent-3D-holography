from GCD_ctrl import *

distance_x = 0
distance_y = 0
G = GCD(1, 2)
while True:
    distance_x = input("distance_x = (m)")
    if(distance_x == ""):
        distance_x = 0
    elif(distance_x == "q"):
        break   
    
    distance_x = float(distance_x)
    distance_x = (distance_x - 0.20) / 0.029 * 29 + 5
    # distance_y = input("distance_y = (mm)")
    # if(distance_y == ""):
    #     distance_y = 0
    print("Linear Motor Stage distance = ", distance_x, "mm")
    G.move_x(distance_x)

G.close()
    
    

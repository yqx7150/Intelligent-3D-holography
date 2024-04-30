import serial.tools.list_ports
import serial
import time
import struct

def sendCommand(ser, command):
    ser.write(command)
    while True:
        if(ser.in_waiting >= 10):
            ser.flushInput()
            break
    print("Command sent: " + ' '.join(format(x, '02x') for x in command))
    
def dir_ctrl(ser, device, dir):  #* 0:前，上  1:后，下
    send_data = bytearray([0x00, 0x00, 0x40, device, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00])
    if(dir == 1):
        send_data[6] = 0x01
    sendCommand(ser, send_data)

def direction_ctrl(ser, device, dir):  
    if(dir == "qian" or dir == "shang"):
        dir_ctrl(ser, device, 0)
    else:
        dir_ctrl(ser, device, 1)
    
def speed_ctrl(ser, device, speed):
    send_data = bytearray([0x00, 0x00, 0x40, device, 0x53, 0x00, 0x00, 0x00, 0x00, 0x00])
    hex_data = struct.pack('!I', speed)
    send_data[6] = hex_data[3]
    send_data[7] = hex_data[2]
    send_data[8] = hex_data[1]
    send_data[9] = hex_data[0]
    sendCommand(ser, send_data)
    
def qianhou_num(ser, device, distance):  #* 1mm = 800个脉冲
    send_data = bytearray([0x00, 0x00, 0x40, device, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00])
    num = round(distance * 800)
    hex_data = struct.pack('!I', num)
    send_data[6] = hex_data[3]
    send_data[7] = hex_data[2]
    send_data[8] = hex_data[1]
    send_data[9] = hex_data[0]
    sendCommand(ser, send_data)
    
def shangxia_num(ser, device, distance):  #* 1mm = 6000个脉冲
    send_data = bytearray([0x00, 0x00, 0x40, device, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00])
    num = round(distance * 6000)
    hex_data = struct.pack('!I', num)
    send_data[6] = hex_data[3]
    send_data[7] = hex_data[2]
    send_data[8] = hex_data[1]
    send_data[9] = hex_data[0]
    sendCommand(ser, send_data)
    
def move_single(ser, device):
    send_data = bytearray([0x00, 0x00, 0x40, device, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00])
    sendCommand(ser, send_data)
    

def move_both(ser, device_qianhou, device_shangxia):
    send_data1 = bytearray([0x00, 0x00, 0x40, device_qianhou, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00])
    send_data2 = bytearray([0x00, 0x00, 0x40, device_shangxia, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00])
    ser.write(send_data1)
    ser.write(send_data2)
    while True:
        if(ser.in_waiting >= 20):
            ser.flushInput()
            break
    
def init_device(ser, device_qianhou, device_shangxia):
    direction_ctrl(ser, device_qianhou, "qian")
    # direction_ctrl(ser, device_shangxia, "xia")
    speed_ctrl(ser, device_qianhou, 13)
    # speed_ctrl(ser, device_shangxia, 15)
    qianhou_num(ser, device_qianhou, 60)
    # shangxia_num(ser, device_shangxia, 20)
    # move_both(ser, device_qianhou, device_shangxia)
    move_single(ser, device_qianhou)
class GCD():
    def __init__(self, qianhou, shangxia) -> None:
        self.qianhou = qianhou
        self.shangxia = shangxia
        
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
        else:
            for port, desc, hwid in sorted(ports):
                if("CP2102 USB to UART Bridge Controller - CP2102 USB to UART Bridge Controller" in desc):
                    print("Found GCD on port: " + port)
                    print("Opening port...")
                    self.ser = serial.Serial(port, 9600)
                    print("Port opened successfully.")
                    
        init_device(self.ser, qianhou, shangxia)
        self.x = 0.
        self.y = 0
        
    def move_x(self, x):
        if(x > self.x):
            direction_ctrl(self.ser, self.qianhou, "hou")
        elif(x < self.x):
            direction_ctrl(self.ser, self.qianhou, "qian")
        else :
            return
            
        qianhou_num(self.ser, self.qianhou, abs(x - self.x))
        move_single(self.ser, self.qianhou)
        self.x = x
        
    def move_y(self, y):
        if(y > self.y):
            direction_ctrl(self.ser, self.shangxia, "shang")
        elif(y < self.y):
            direction_ctrl(self.ser, self.shangxia, "xia")
        else :
            return
            
        shangxia_num(self.ser, self.shangxia, abs(y - self.y))
        move_single(self.ser, self.shangxia)
        self.y = y
        
    def move_xy(self, x, y):
        if(x > self.x):
            direction_ctrl(self.ser, self.qianhou, "hou")
        elif(x < self.x):
            direction_ctrl(self.ser, self.qianhou, "qian")
        qianhou_num(self.ser, self.qianhou, abs(x - self.x))
        
        if(y > self.y):
            direction_ctrl(self.ser, self.shangxia, "shang")
        elif(y < self.y):
            direction_ctrl(self.ser, self.shangxia, "xia")
        shangxia_num(self.ser, self.shangxia, abs(y - self.y))
        
        move_both(self.ser, self.qianhou, self.shangxia)
        
        self.x = x
        self.y = y
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def close(self):
        self.ser.close()
        
        
if  __name__ == "__main__":        
    G = GCD(1, 2)
    G.move_xy(10, 10)
    time.sleep(5)
    G.move_xy(5.5, 5.5)

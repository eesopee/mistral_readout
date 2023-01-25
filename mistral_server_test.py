import subprocess as sp
import time
#import client_lib

while True:

    
    command = raw_input("Insert command to run: ")
    
    if command == "initialize":
        
        print("Intialize")
        
        client = sp.Popen(["python2", "./mistral_client_current.py","norun"], stdin=sp.PIPE, stdout=sp.PIPE, shell=False)
        status = sp.Popen.poll(client)
        client.stdin.write('0\n')

    if command == "start":
        
        client.stdin.write("15\n")

        if status == None:
            print("process started at timestamp ", time.time())

    if command == "stop":

        client.stdin.write("^C\n")
        print("process terminated at timestamp", time.time())

    if command == "target-sweep":

        #targetSweepProcess = sp.Popen(['python',''])
        #status = sp.Popen.poll(targetSweepProcess)

        if status == None:
            print("Target sweep process started at timestamp ", time.time())

        client.stdin.write("7\n")
        client.stdin.write("y")
        #client.stdin.write("/home/mistral/src/parameters/cosmo_singlepixel_1\n")
        #client.stdin.write("\n")

        
    if command == "exit":

        client.communicate(input=b"16")

        quit()




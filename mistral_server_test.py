import subprocess as sp
import time
import client_lib

while True:

    command = input("Insert command to run: ")

    if command == "initialize":
        print("Intiialize")
	ri = client_lib.roachInterface()
        

    if command == "start":

        acquireProcess = sp.Popen(['python','mistral_client_current.py current'])
        status = sp.Popen.poll(acquireProcess)

        if status == None:
            print("process started at timestamp ", time.time())

    if command == "stop":

        sp.Popen.terminate(acquireProcess)

        print("process terminated at timestamp", time.time())

    if command == "target-sweep":

        targetSweepProcess = sp.Popen(['python',''])
        status = sp.Popen.poll(targetSweepProcess)

        if status == None:
            print("Target sweep process started at timestamp ", time.time())

    if command == "exit":
        quit()


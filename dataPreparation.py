
import os
import shutil


closedEyesDir = os.path.join("MRL_DATA", "Closed_Eyes")
openEyesDir = os.path.join("MRL_DATA", "Open_Eyes")

os.makedirs(closedEyesDir, exist_ok=True)
os.makedirs(openEyesDir, exist_ok=True)

rawDataDir = 'mrlEyes_2018_01'
for dirPath, dirName, fileNames in os.walk(rawDataDir):
    for i in [f for f in fileNames if f.endswith('.png')]:
        if i.split('_')[4] == 0:
            shutil.copy(src=dirPath + '/' + i, dst=closedEyesDir)
        elif i.split('_')[4] == '1':
            shutil.copy(src=dirPath + '/' + i, dst=openEyesDir)

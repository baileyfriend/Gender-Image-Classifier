# Using a genderize script from Philip Masek
# Takes in names from the Labeled Faces in the Wild dataset, adds them to
# male or female folder based on the Genderize.io api which guesses gender
# based on name.
# Only classifies as male or female if over 90% sure that name belongs to 
# a sepecific gender.

import os
import sys
from django.http import request
from os import rename
import requests
import shutil

__author__ = 'Philip Masek'



def main(argv):
    fileList = []
    fileSize = 0
    folderCount = 0
    rootdir = "/Users/baileyfreund/Desktop/AI/proj3/lfw/"
    maleFolder = "/Users/baileyfreund/Desktop/AI/proj3/male/"
    femaleFolder = "/Users/baileyfreund/Desktop/AI/proj3/female/"
    count = 0
    tmp = ""

    for root, subFolders, files in os.walk("/Users/baileyfreund/Desktop/AI/proj3/lfw/"):
        folderCount += len(subFolders)
        for file in files:
            if not file.startswith('.') and os.path.isfile(os.path.join(root,file)):
                f = os.path.join(root,file)
                fileSize = fileSize + os.path.getsize(f)
                fileSplit = file.split("_")
                fileList.append(f)
                count += 1
            

                if count == 1:
                    result = requests.get("https://api.genderize.io?name=%s" % fileSplit[0])
                    result = result.json()
                    print(result)
                    tmp = fileSplit[0]
                elif tmp != fileSplit[0]:
                        result = requests.get("https://api.genderize.io?name=%s" % fileSplit[0])
                        result = result.json()
                        tmp = fileSplit[0]
                else:
                    tmp = fileSplit[0]

                try:
                    if float(result['probability']) > 0.9:
                        if result['gender'] == 'male':
                            shutil.copyfile(f,"%s/%s" % (maleFolder,file))
                        elif result['gender'] == 'female':
                            shutil.copyfile(f,"%s/%s" % (femaleFolder,file))
                except Exception as e:
                    #print(result['name'])
                    print("Exception Found")
                    
                print(count)



if __name__ == "__main__":
    main(sys.argv)
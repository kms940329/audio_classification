import subprocess

command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format("/home/kms/nlp/NLP_project_v2/data/siren.mp4", "siren.wav")

subprocess.call(command, shell=True)

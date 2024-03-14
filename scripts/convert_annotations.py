import ffmpeg
import json 
import pandas as pd 
import os 
import random 

import argparse 
import cv2
import os
import subprocess32 as sp
import json


def probe(vid_file_path):
	''' Give a json from ffprobe command line

	@vid_file_path : The absolute (full) path of the video file, string.
	'''
	if type(vid_file_path) != str:
		raise Exception('Gvie ffprobe a full file path of the video')
		return

	command = ["ffprobe",
			"-loglevel",  "quiet",
			"-print_format", "json",
			 "-show_format",
			 "-show_streams",
			 vid_file_path
			 ]

	pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT)
	out, err = pipe.communicate()
	return json.loads(out)

def compute_duration(vid_file_path):
	''' Video's duration in seconds, return a float number
	'''
	_json = probe(vid_file_path)

	if 'format' in _json:
		if 'duration' in _json['format']:
			return float(_json['format']['duration'])

	if 'streams' in _json:
		# commonly stream 0 is the video
		for s in _json['streams']:
			if 'duration' in s:
				return float(s['duration'])

	# if everything didn't happen,
	# we got here because no single 'return' in the above happen.
	raise Exception('I found no duration')

	

parser=argparse.ArgumentParser()
parser.add_argument('--actionfile')
parser.add_argument('--csvpath')
parser.add_argument('--jsonpath')
parser.add_argument('--videopath')

args=parser.parse_args()


finalcsvdir=args.csvpath 
jsondir=args.jsonpath 
videodir=args.videopath 



#with open('scripts/pinwheelactions.txt','r') as f:
with open(args.actionfile,'r') as f:
	actionnames = f.read().splitlines()
#print(actionnames)


actioncolors= ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(actionnames)+1)]


jsonactions=[]
jsonactions.append({"id":0,"name":"default","color":str(actioncolors[0]),"objects":[0]})




for idx,a in enumerate(actionnames):
	jsonactions.append({"id":idx+1,"name":str(a),"color":str(actioncolors[idx+1]),"objects":[0]})
	


for csvfile in os.listdir(finalcsvdir):
	# if '19' in csvfile:
	# 	continue

	actionannotations=[]
	
	data=pd.read_csv(os.path.join(finalcsvdir,csvfile),sep=',')
	
	videofile='Video'+csvfile.replace('download','').replace('.csv','.mp4')
	# print(videofile)
	# info=ffmpeg.probe(videofile)
	# v=f"{info['format']['duration']}"

	# duration=float(v)
	
	if not (videofile.endswith('mp4') or videofile.endswith('.mpeg')):
		print('Video file not ending in mp4 or mpeg. Continuing to the next video.')
		continue
	
	videofilepath = os.path.join(videodir, videofile)
	

	if not os.path.exists(videofilepath):
		print(videofilepath,'does not exist')
		continue
	
	print(videofilepath)
	duration = compute_duration(videofilepath)
	
	#duration=videodurations.loc[videodurations.iloc[:,0] == videofile,:].values[0][1]


	#print(data.iloc[1,0],data.shape,duration)
	assert ('Place' in data.iloc[1,0])
	
	tempdata=data.iloc[1:,1:3]
	#print(data.iloc[1:,1:3])
	for j in range(tempdata.shape[0]):
		actionannotations.append({"start":float(tempdata.iloc[j,0]),"end":float(tempdata.iloc[j,1]),"action":j+1,"object":0,"color":actioncolors[j],"description":""})
	
	#print(len(actionannotations))


	jsondata={"version":"2.0.3","annotation":{"video":{"src":"blob:https://vidat2.davidz.cn/766b023f-7282-4982-9d52-737858089abc","fps":10,"frames":int(duration*10),"duration":duration,"height":720,"width":1280},
										  "keyframeList":[],"objectAnnotationListMap":{},"regionAnnotationListMap":{},"skeletonAnnotationListMap":{},
										  "actionAnnotationList":actionannotations},
		  "config":{"objectLabelData":[{"id":0,"name":"default","color":"#00FF00"}],"actionLabelData":jsonactions,"skeletonTypeData":[]}}
	
	with open(os.path.join(jsondir,csvfile.replace('.csv','.json')), 'w', encoding='utf-8') as f:
		json.dump(jsondata, f)
	
	

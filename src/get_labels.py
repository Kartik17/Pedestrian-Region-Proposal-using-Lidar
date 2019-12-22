import os
if __name__ == '__main__':
	labels = os.listdir('./Data/label')
	file_name_list = []
	for label in labels:
		file = open(os.path.join('./Data/label',str(label)))
		lines = file.readlines()
		counter = 0
		for line in lines:
			class_id = line.split(' ')[0]
			if class_id == 'Pedestrian':
				counter += 1
			else:
				pass
		if counter > 4:
			file_name_list.append(label)
	with open('./get_labels.txt','w+') as f:
		for file_name in file_name_list:
			f.write(str(file_name).split('.')[0]+'\n')
	f.close()
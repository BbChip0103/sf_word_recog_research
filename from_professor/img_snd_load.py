def img_snd_load():

	X_train = np.empty((0,np.prod(img_dim)))
	y_train = np.empty(0, dtype=int)
	with open(train_list, 'rt') as filelist:
	    idx = 0
	    for fname in filelist.read().splitlines():
		idx += 1
		img = np.array(Image.open(os.path.join(sound_path,fname)))
		if img.shape == img_dim:
		    X_train = np.append(X_train, img.reshape(1,-1), axis=0)  
		    y_train = np.append(y_train, \
		       [i for i in range(len(list_words)) if os.path.split(fname)[0] == list_words[i]])
		if idx % 100 == 0:    
		    print(idx)
		    
	X_validation = np.empty((0,np.prod(img_dim)))
	y_validation = np.empty(0, dtype=int)
	with open(validation_list, 'rt') as filelist:
	    idx = 0
	    for fname in filelist.read().splitlines():
		idx += 1
		img = np.array(Image.open(os.path.join(sound_path,fname)))
		if img.shape == img_dim:
		    X_validation = np.append(X_validation, img.reshape(1,-1), axis=0)  
		    y_validation = np.append(y_validation, \
		       [i for i in range(len(list_words)) if os.path.split(fname)[0] == list_words[i]])
		if idx % 100 == 0:    
		    print(idx)
		    
	X_test = np.empty((0,np.prod(img_dim)))
	y_test = np.empty(0, dtype=int)
	with open(test_list, 'rt') as filelist:
	    idx = 0
	    for fname in filelist.read().splitlines():
		idx += 1
		img = np.array(Image.open(os.path.join(sound_path,fname)))
		if img.shape == img_dim:
		    X_test = np.append(X_test, img.reshape(1,-1), axis=0)  
		    y_test = np.append(y_test, \
		       [i for i in range(len(list_words)) if os.path.split(fname)[0] == list_words[i]])
		if idx % 100 == 0:    
		    print(idx)

	return X_train, y_train, X_validation, y_validatoin, X_test, y_test

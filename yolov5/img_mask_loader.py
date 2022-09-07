import numpy as np
import os
import glob
import cv2

EXT = ['jpg', 'png', 'jpeg']

def read_data(data_dir, image_size, no_label=False):

    im_paths = []

    for x in EXT:
        im_paths.extend(glob.glob(os.path.join(data_dir, 'images', '*.{}'.format(x)))) # 얘는 나중에 설정하자.

    imgs = []
    labels = []

    for im_path in im_paths: # 경로 리스트 안의 경로마다 --> 공부한게 완전 의미 없는건 아닌듯.
        # 이미지를 load 한다.
        im_name = os.path.splitext(os.path.basename(im_path))[0] # anyimagename.(확장자) 에서 anyimagename 을 im_name에 저장한다.
        im = cv2.imread(im_path) # 이미지를 read 한다.
        im = cv2.resize(im, (224, 224)) # resize
        imgs.append(im) # imgs 리스트에 im 를 저장한다.

        if no_label:
            labels.append(0) # 라벨이 없다면 labels 리스트에 0을 append 한다.
            continue

        # mask load
        mask_path = os.path.join(data_dir, 'masks', '{}.png'.format(im_name)) # {}가 뭐지? 어느 png든 다 넣는다는 것인듯.
        mask = cv2.imread(mask_path) # mask를 불러들임.
        mask = cv2.resize(mask, (224, 224)) # resize

        label = np.zeros((image_size[0], image_size[1], 3), dtype=np.float16) # 라벨 초기화 (이미지 사이즈만큼, 3채널)
        label.fill(-1) # 왜 -1로 채워넣지?
        # Pixel annotations 1:Foreground, 2:Background, 3:Unknown --> 나는 1, 2, 3, 4, 5로 만들고, 밑의 if문을 수정하자.
        # Pixel annotations 1:Background, 2:(), 3:(), 4:(), 5:()
        idx = np.where(mask == 1) # mask에서 2인 idx들을 반환
        label[idx[0],idx[1],:] = [1, 0, 0, 0, 0] # 아하, idx[0]는 width, idx[1]은 height 구나

        idx = np.where(mask == 1)
        if : # mask에서 1이 있다면
            label[idx[0],idx[1],:] = [0, 1, 0, 0, 0]
        if : # mask에서 1이 있다면
            label[idx[0],idx[1],:] = [0, 0, 1, 0, 0]
        if : # mask에서 1이 있다면
            label[idx[0],idx[1],:] = [0, 0, 0, 1, 0]
        if : # mask에서 1이 있다면
            label[idx[0],idx[1],:] = [0, 0, 0, 0, 1]    
        # 그렇다면 나는 [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], ... , [0, 0, 0, 0, 1] --> 일단 만들어는 놓자. 써먹을데가 있을 수 도 있다.
        # 어떤 종류의 파손이 있는지 띄울수 도 있다고 설명해야겠다. --> 위의 코드가 픽셀마다 넣는건가, 아니면 이미지 이름마다 넣는건가.
        labels.append(label)

    X_set = np.array(imgs, dtype=np.float16) # --> 이미지를 np.array로 반환 (float16으로 만들면 학습률이 떨어질까? --> float32가 학계에서는 주요 dtype인 것 같다.)
    y_set = np.array(labels, dtype=np.float16) # --> label을 np.array로 반환 --> 맞는듯? 픽셀마다 라벨링 하는거

    return X_set, y_set # return mask 하면 mask return 할 수 있다.